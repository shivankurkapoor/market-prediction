'''
This class contains classes for various deep neural net models
'''

import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers.python.layers.initializers import xavier_initializer


class RNNModel():
    '''RNN (GRU cells) classifier with attention mechanism'''

    def __init__(self, batch_size, num_features, rnn_hidden_size, attn_len, num_classes, beta, lr):
        global_step = tf.train.get_or_create_global_step()
        self.batch_size = batch_size
        self.num_features = num_features
        self.rnn_hidden_size = rnn_hidden_size
        self.attn_len = attn_len
        self.num_classes = num_classes
        self.beta = beta
        self.lr = lr
        self.input_data = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.num_features])
        self.target_data = tf.placeholder(dtype=tf.int32, shape=[self.batch_size])
        self.dropout_prob = tf.placeholder(dtype=tf.float32, shape=[])
        self.gru_cell = self.makeGRUCells()
        self.zero_state = self.gru_cell.zero_state(1, tf.float32)
        self.start_state = tf.placeholder(dtype=tf.float32, shape=[1, self.gru_cell.state_size])

        with tf.variable_scope("ff", initializer=xavier_initializer(uniform=False)):
            dropped_input = tf.nn.dropout(self.input_data, keep_prob=self.dropout_prob)

        split_inputs = tf.reshape(dropped_input, shape=[1, self.batch_size, self.num_features],
                                  name="reshape_l1")  # Each item in the batch is a time step, iterate through them
        split_inputs = tf.unstack(split_inputs, axis=1, name="unpack_l1")
        states = []
        outputs = []
        with tf.variable_scope("rnn", initializer=xavier_initializer(uniform=False)) as scope:
            state = self.start_state
            for i, inp in enumerate(split_inputs):
                if i > 0:
                    scope.reuse_variables()

                output, state = self.gru_cell(inp, state)
                states.append(state)
                outputs.append(output)
        self.end_state = states[-1]
        outputs = tf.stack(outputs, axis=1)  # Pack them back into a single tensor
        outputs = tf.reshape(outputs, shape=[self.batch_size, self.rnn_hidden_size])
        self.logits = tf.contrib.layers.fully_connected(
            num_outputs=self.num_classes,
            inputs=outputs,
            activation_fn=None
        )

        with tf.variable_scope("loss"):
            self.penalties = tf.reduce_sum([self.beta * tf.nn.l2_loss(var) for var in tf.trainable_variables()])

            self.losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.target_data)
            self.loss = tf.reduce_sum(self.losses + self.beta * self.penalties)

        with tf.name_scope("train_step"):
            opt = tf.train.AdamOptimizer(self.lr)
            gvs = opt.compute_gradients(self.loss)
            self.train_op = opt.apply_gradients(gvs, global_step=global_step)

        with tf.name_scope("predictions"):
            probs = tf.nn.softmax(self.logits)
            self.predictions = tf.argmax(probs, 1)
            correct_pred = tf.cast(tf.equal(self.predictions, tf.cast(self.target_data, tf.int64)), tf.float64)
            self.accuracy = tf.reduce_mean(correct_pred)

        # Create a summary for cost and accuracy
        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("accuracy", self.accuracy)

        # merge all summaries into a single "operation" which we can execute in a session
        self.summary_op = tf.summary.merge_all()

    def makeGRUCells(self):
        cells = []
        global NUM_LAYERS
        NUM_LAYERS = NUM_LAYERS or 2
        for _ in range(NUM_LAYERS):
            base_cell = tf.nn.rnn_cell.GRUCell(num_units=self.rnn_hidden_size)
            attn_cell = tf.contrib.rnn.AttentionCellWrapper(cell=base_cell, attn_length=self.attn_len,
                                                            state_is_tuple=False)
            cells.append(attn_cell)
        layered_cell = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=False)
        return layered_cell


class DARNNModel(object):
    '''
    Encoder-Decoder model (LSTM cells) with attention mechanism
    Ref - https://arxiv.org/pdf/1704.02971.pdf
    '''

    def __init__(self, input_dim, time_step, e_hidden, d_hidden, o_hidden, batch_size):
        self.batch_size = batch_size
        self.e_hidden = e_hidden
        self.d_hidden = d_hidden
        self.o_hidden = o_hidden
        self.input_dim = input_dim
        self.time_step = time_step
        self.seq_len = tf.placeholder(tf.int32, [None])

        # Creating placeholders for input x, input y, labels
        # Input X -> batch_size * T * d
        # Input Y -> batch_size * T (target series)
        # n are the number of driving series
        # T is the window size
        self.input_x = tf.placeholder(dtype=tf.float32, shape=[None, None, self.input_dim], name='Input_X')
        self.input_y = tf.placeholder(dtype=tf.float32, shape=[None, self.time_step], name='Input_Y')
        self.label = tf.placeholder(dtype=tf.float32)

        # Creating cells for encoder, decoder
        self.encode_cell = tf.contrib.rnn.LSTMCell(self.e_hidden, forget_bias=1.0, state_is_tuple=True)
        self.decode_cell = tf.contrib.rnn.LSTMCell(self.d_hidden, forget_bias=1.0, state_is_tuple=True)
        self.output_cell = tf.contrib.rnn.LSTMCell(self.o_hidden, forget_bias=1.0, state_is_tuple=True)

        # Initializing loss
        self.loss = tf.constant(0.0, name='Loss')

        ##############################################################################################
        # Building the model - https://arxiv.org/pdf/1704.02971.pdf
        #############################################################################################

        ##############ENCODER##############
        out = self.encoder_rnn(self.input_x)
        self.out = out  # b * T * 2*e_hidden
        out = tf.transpose(out, [0, 2, 1], name='transpose_line_36')  # b * 2*e_hidden *T
        with tf.name_scope('encoder') as scope:
            stddev = 1.0 / (self.e_hidden * self.time_step)
            Ue = tf.Variable(dtype=tf.float32,
                             initial_value=tf.truncated_normal(shape=[self.time_step, self.time_step],
                                                               mean=0.0, stddev=stddev),
                             name='Ue')
        var = tf.tile(tf.expand_dims(Ue, 0), [self.batch_size, 1, 1], name='tile_Ue')  # b*T*T
        batch_mul = tf.matmul(var, self.input_x,
                              name='matmul_line44')  # (b*T*T)*(b*T*d) = (b,T,d)--UeXk component of Eq8, d is number of dims
        self.out = batch_mul
        e_list = []

        for k in range(self.input_dim):
            series_k = tf.reshape(batch_mul[:, :, k], [self.batch_size, self.time_step, 1],
                                  name='reshape_line49')  # b T 1
            e_k = self.attention(out, series_k, scope='encoder')  # calculating eq 8
            e_list.append(e_k)
        e_list = tf.concat(e_list, axis=1)
        soft_attention = tf.nn.softmax(e_list, dim=1, name='softmax_line53')  # b d T
        input_attention = tf.multiply(self.input_x, tf.transpose(soft_attention, [0, 2, 1]),
                                      name='multiply_line54')  # b T d

        with tf.variable_scope('fw_lstm') as scope:
            tf.get_variable_scope().reuse_variables()
            h, _ = tf.nn.dynamic_rnn(self.encode_cell, input_attention, self.seq_len,
                                     dtype=tf.float32)
            # h : b T eh

        #############DECODER############
        d, dec_out = self.decoder_rnn(h)  # d: b, T ,dh dec_out: b T 2dh
        self.out = d
        dec_out = tf.transpose(dec_out, [0, 2, 1], name='transpose_line65')
        with tf.name_scope('decoder') as scope:
            stddev = 1. / (self.d_hidden * self.time_step)  ## it should be e_hidden * time_step
            Ud = tf.Variable(dtype=tf.float32,
                             initial_value=tf.truncated_normal(shape=[self.e_hidden,
                                                                      self.e_hidden],
                                                               mean=0.0,
                                                               stddev=stddev,
                                                               ), name='Ud')

        de_var = tf.tile(tf.expand_dims(Ud, 0), [self.batch_size, 1, 1], name='tile_line76')  # b, eh, eh
        batch_mul_de = tf.matmul(h, de_var, name='matmul_line77')  # b T eh * b eh eh = b T eh
        batch_mul_de = tf.transpose(batch_mul_de, [0, 2, 1], name='transpose_line78')  # b eh T
        e_de_list = []
        for t in range(self.time_step):
            series_t = tf.reshape(batch_mul_de[:, :, t], [self.batch_size, self.e_hidden, 1], name='reshape_line81')
            e_t = self.de_attention(dec_out, series_t, scope='decoder')
            e_de_list.append(e_t)
        e_de_list = tf.concat(e_de_list, axis=1, name='concat_line84')  # b T T
        de_soft_attention = tf.nn.softmax(e_de_list, dim=1, name='softmax_line85')  # sames dims as e_de_list

        #######################CONTEXT#########################
        c_list = []
        for t in range(self.time_step):
            Beta_t = tf.expand_dims(de_soft_attention[:, :, t], -1, name='expand_line90')  # b T 1
            weighted = tf.reduce_sum(tf.multiply(Beta_t, h), 1,
                                     name='reducesum_line91')  # b T 1 . b T eh = reducesum(b T eh) = b eh
            c_list.append(tf.expand_dims(weighted, 1, name='expand_line92'))  # b 1 eh
        c_t = tf.concat(c_list, axis=1, name='concat_line93')  ## b T eh
        self.out = c_t
        # concat - b T 1 & b T eh = b T eh+1
        c_t_hat = tf.concat([tf.expand_dims(self.input_y, -1), c_t], axis=2,
                            name='concat_line95')  # b, T, (eh+1) where +1 is for concatenation

        ####################Y_Hat##############################

        with tf.variable_scope('temporal'):
            mean = 0.0
            stddev = 1. / (self.e_hidden * self.time_step)
            W_hat = tf.get_variable(name='W_hat', shape=[self.e_hidden + 1, 1], dtype=tf.float32,
                                    initializer=tf.truncated_normal_initializer(mean, stddev))
        W_o = tf.tile(tf.expand_dims(W_hat, 0), [self.batch_size, 1, 1], name='tile_line104')
        y_hat = tf.matmul(c_t_hat, W_o, name='matmul_line105')  # b T 1+eh X b 1+eh 1 = b T 1

        ################Final Step###########################

        d_y_concat = tf.concat([d, y_hat], axis=2, name='concat_line110')  ## b T dh+1
        with tf.variable_scope('out_lstm') as scope:
            d_final, _ = tf.nn.dynamic_rnn(self.output_cell, d_y_concat, self.seq_len, dtype=tf.float32)  # b T o_hidden

        ##############Output Y_t ##########################
        d_c_concat = tf.concat([d_final[:, -1, :], c_t[:, -1, :]], axis=1, name='concat_line116')  # b, oh+eh
        d_c_concat = tf.expand_dims(d_c_concat, -1, name='expand_line117')  # b,oh+eh,1

        with tf.variable_scope('predict'):
            mean = 0.0
            stddev = 1.0 / (self.e_hidden * self.time_step)
            Wy = tf.get_variable(name='Wy', shape=[self.o_hidden, self.o_hidden + self.e_hidden], dtype=tf.float32,
                                 initializer=tf.truncated_normal_initializer(mean, stddev))
            Vy = tf.get_variable(name='Vy', shape=[self.o_hidden], dtype=tf.float32,
                                 initializer=tf.truncated_normal_initializer(mean, stddev))
            bw = tf.get_variable(name='bw', shape=[self.o_hidden], dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        W_y = tf.tile(tf.expand_dims(Wy, 0), [self.batch_size, 1, 1], name='tile_line128')  # b,oh,oh+eh
        b_w = tf.expand_dims(tf.tile(tf.expand_dims(bw, 0), [self.batch_size, 1]), -1,
                             name='expand_line129')  # b,oh -> b,oh,1
        V_y = tf.tile(tf.expand_dims(Vy, 0), [self.batch_size, 1], name='tile_line130')  # b,oh
        V_y = tf.expand_dims(V_y, 1, name='expand_line131')  # b,1,oh
        self.y_predict = tf.squeeze(tf.matmul(V_y, tf.matmul(W_y, d_c_concat) + b_w),
                                    name='squeeze_line132')  # (b,1,oh) * (b,oh,1) -> squeeze -> (b,)

        self.loss += tf.reduce_mean(tf.square(self.label - self.y_predict), name='reduce_mean_line134')
        self.params = tf.trainable_variables()
        optimizer = tf.train.AdamOptimizer(1e-3)
        # self.train_op = optimizer.minimize(self.loss)  ~ compute_gradients + apply_gradients
        grad_var = optimizer.compute_gradients(loss=self.loss, var_list=self.params, aggregation_method=2)
        self.train_op = optimizer.apply_gradients(grad_var, name='apply_gradients_line139')

    def encoder_rnn(self, input_x):
        with tf.variable_scope('fw_lstm') as scope:
            # In Dynamic RNN we get a dynamic graph based on the input length
            # This is known to be as fast as it's static counterpart
            # Since our encoded cell is an LSTM cell, it return a LSTMStateTuple (ct, mt)
            # c -> cell state of LSTM cell
            # m ->  output of LSTM cell
            # out -> Output of the unrolled RNN ([batch_size * T * hidden_units])
            # states -> 'state' is a N-tuple, where N is the number of LSTMCells, containing a
            # tf.contrib.rnn.LSTMStateTuple for each cell - (c, h) c = hidden state, h = output of LSTM (or c, m)
            # here we only have a single state because there is just 1 LSTM cell passes into dynamic_rnn
            out, state = tf.nn.dynamic_rnn(self.encode_cell, input_x, self.seq_len, dtype=tf.float32)
            state = tf.Print(state, [state, tf.shape(state)], "These are the fucking states")
        tmp = tf.tile(state[0], [1, self.time_step], name='tile_encoder_rnn')
        tmp = tf.reshape(tmp, [self.batch_size, self.time_step, self.e_hidden], name='reshape_encoder_rnn')
        concat = tf.concat([out, tmp], axis=2, name='concat_encoder_rnn')
        # According to the paper - out -> h and tmp -> s
        # So my interpretation is that out is actaully the output of the LSTM i.e. m in LSTM paper
        # or h in tensorflow or h in DARNN paper. And state is the (c,h) or (s,h) --- so shouldn't we use
        # state[0]
        return concat  ## shape should be (b, T, 2*e_hidden)

    def decoder_rnn(self, h):
        with tf.variable_scope('dec_lstm') as scope:
            cell = self.decode_cell
            d, s = tf.nn.dynamic_rnn(self.decode_cell, h, self.seq_len, dtype=tf.float32)
        tmp = tf.tile(s[0], [1, self.time_step], name='tile_decoder_rnn')
        tmp = tf.reshape(tmp, [self.batch_size, self.time_step, self.d_hidden], name='reshape_decoder_rnn')
        concat = tf.concat([d, tmp], axis=2, name='concat_decoder_rnn')  # b T 2*d_hidden
        return d, concat

    def attention(self, out, series_k, scope=None):
        with tf.variable_scope('encoder') as scope:
            try:
                mean = 0.
                stddev = 1. / (self.e_hidden * self.time_step)
                # See Eq 8
                We = tf.get_variable(name='We', dtype=tf.float32, shape=[self.time_step, 2 * self.e_hidden],
                                     initializer=tf.truncated_normal_initializer(mean, stddev))
                Ve = tf.get_variable(name='Ve', dtype=tf.float32, shape=[1, self.time_step],
                                     initializer=tf.truncated_normal_initializer(mean, stddev))
            except ValueError:
                scope.reuse_variables()
                We = tf.get_variable('We')
                Ve = tf.get_variable('Ve')
        We = tf.tile(tf.expand_dims(We, 0), [self.batch_size, 1, 1], name='tile_attention')
        brcast = tf.nn.tanh(tf.matmul(We, out) + series_k,
                            name='tanh_attention')  # b * T * 2eh X b * 2eh * T + b * T * 1 = b T T
        V_e = tf.tile(tf.expand_dims(Ve, 0), [self.batch_size, 1, 1], name='tile_attention')  # b 1 T
        return tf.matmul(V_e, brcast, name='matmul_attention')  # b 1 T

    def de_attention(self, out, series_k, scope='None'):
        with tf.variable_scope('decoder') as scope:
            try:
                mean = 0.
                stddev = 1.0 / (self.d_hidden * self.time_step)
                Wd = tf.get_variable(name='Wd', dtype=tf.float32, shape=[self.e_hidden, 2 * self.d_hidden],
                                     initializer=tf.truncated_normal_initializer(mean, stddev))
                Vd = tf.get_variable(name='Vd', dtype=tf.float32, shape=[1, self.e_hidden],
                                     initializer=tf.truncated_normal_initializer(mean, stddev))

            except ValueError as e:
                scope.reuse_variables()
                Wd = tf.get_variable('Wd')
                Vd = tf.get_variable('Vd')
        W_d = tf.tile(tf.expand_dims(Wd, 0), [self.batch_size, 1, 1], name='Wd_tile_de_attention')
        brcast = tf.nn.tanh(tf.matmul(W_d, out) + series_k,
                            name='tanh_de_attention')  # b eh 2dh * b 2dh T + b eh 1(T) = b eh T
        V_d = tf.tile(tf.expand_dims(Vd, 0), [self.batch_size, 1, 1], name='Vd_tile_de_attention')  # b 1 eh
        return tf.matmul(V_d, brcast, name='matmul_de_attention')  # b 1 eh * b eh T = b 1 T

    def predict(self, x_test, y_test, sess):
        train_seq_len = np.ones(self.batch_size) * self.time_step
        feed = {
            self.input_x: x_test,
            self.input_y: y_test,
            self.seq_len: train_seq_len,
        }
        y_hat = sess.run(self.y_predict, feed_dict=feed)
        return y_hat
