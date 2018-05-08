"""

"""

import urllib
import quandl
import pandas as pd
from common.consts import *
from common.utils import http_request
from marketdata.financedata import FinanceData
from marketdata.financedata import validate_params


class QuandlAPI(FinanceData):

    def __init__(self, finance_data_type):
        super(QuandlAPI, self).__init__(finance_data_type)

    @property
    def API(self):
        return Q_API

    @validate_params
    def download_csv(self, **kwargs):
        """

        :param start_date: Retrieve data rows on and after the specified start date.
        :param end_date: Retrieve data rows up to and including the specified end date.
        :param dataset_code: Code identifying the dataset.
        :param collapse: Options are daily, weekly, monthly, quarterly, annual
        :param limit: Use limit=n to get the first n rows of the dataset. Use limit=1 to get just the
                      latest row.
        :param order: Return data in ascending or descending order of date. Default is desc.
        :param transform: Perform elementary calculations on the data prior to downloading. Default is none.
                          Options are diff, rdiff, cumul, and normalize

        :return: data in pandas dataframe format
        """
        kwargs.update({'api_key': QUANDL_API_KEY})
        params = urllib.urlencode(kwargs)
        URL2 = QUANDL_URL2.format(database_code=kwargs.get('database_code'),
                                  dataset_code=kwargs.get('dataset_code'))
        status, data = http_request(url=QUANDL_URL, params=params, ishttps=True, url2=URL2)
        return status, data

    @validate_params
    def download_data(self, *args, **kwargs):
        """
         This method uses quandl api to get data in numpy or pandas format
        :param kwargs:
        :return: pandas dataframe or numpy ndarray
        """
        database_code = kwargs.get('database_code')
        dataset_code = kwargs.get('dataset_code')
        options = dict()
        data = None
        status = INT_ERROR_GENERAL
        options.update({'api_key': QUANDL_API_KEY})
        dataset = '/'.join([database_code, dataset_code])
        if 'start_date' in kwargs:
            options.update({'start_date': kwargs.pop('start_date')})
        if 'end_date' in kwargs:
            options.update({'end_date': kwargs.pop('end_date')})
        if 'collapse' in kwargs:
            options.update({'collapse': kwargs.pop('collapse')})
        if 'transform' in kwargs:
            options.update({'transform': kwargs.pop('transform')})
        if 'order' in kwargs:
            options.update({'order': kwargs.pop('order')})
        if 'rows' in kwargs:
            options.update({'rows': kwargs.pop('rows')})
        if 'returns' in kwargs:
            options.update({'returns': kwargs.pop('returns')})

        try:
            data = quandl.get(dataset, **options)
            if (isinstance(data, pd.DataFrame) and not data.empty) or data:
                status = INT_OK

        except all as e:
            print 'Error in Quandl API call: %s' % (e)
        return status, data
