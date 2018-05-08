"""
This module contains download and other methods for AlphaVantage API
"""
import urllib
import json
import pandas as pd
from common.consts import *
from common.utils import http_request
from marketdata.financedata import FinanceData
from marketdata.financedata import validate_params


class AlphaVantageAPI(FinanceData):

    def __init__(self, finance_data_type):
        super(AlphaVantageAPI, self).__init__(finance_data_type)

    @property
    def API(self):
        return AV_API

    @validate_params
    def download_data(self, *args, **kwargs):
        options = kwargs
        options.update({'apikey': ALPHA_VANTAGE_API_KEY})
        params = urllib.urlencode(options)
        status, data = http_request(url=AV_URL, params=params, ishttps=True, url2=AV_URL2)
        return status, data

    @staticmethod
    def json_to_df(json_str):
        try:
            json_dict = json.loads(json_str)
        except Exception as e:
            print('Not a JSON - {}'.format(e))
            return None
        else:
            data = []
            for k1, v1 in json_dict.iteritems():
                if 'Meta Data' not in k1:
                    for k2, v2 in v1.iteritems():
                        v2.update({'timestamp' : k2})
                        data.append(v2)
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df





