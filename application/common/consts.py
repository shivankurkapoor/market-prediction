"""
This file defines all the global codes like API keys, error codes etc
"""
import re

#######################################################################################################################
# HTTP REQUEST RESPONSE
#######################################################################################################################
LIST_HTTPMETHODS = ['GET', 'POST']

STR_UNDEFINED = 'Undefined'
STR_NOTFOUND = 'Notfound'
INT_ERROR_GENERAL = -1000
INT_ERROR_NOTEXIST = -3000
INT_ERROR_FOUND = -3001
INT_ERROR_PASSEDEXPTIME = -2000
INT_ERROR_FORMAT = -4000
INT_ERROR_TIMEOUT = -4008
INT_ERROR_MAXATTEMPTREACHED = -40080
INT_FAILURE = -1001
INT_FAILURE_AUTH = -1002
INT_ILLEGAL_HTTPMETHOD = -4001
INT_NOTEXISTS = 3000
INT_FOUND = 3001
INT_LOGGEDOUT = 1009
INT_OK = 0
INT_CREATED = 2001

RE_URL = re.compile(r'^[0-9a-z.\-_]{1,50}$')
RE_SUBURL = re.compile(r'^[0-9a-z\\\-_]{1,50}$')

#######################################################################################################################
# QUANDL - stocks prices, stock vol, VIX
#######################################################################################################################
Q_API = 'Quandl API'
QUANDL_DATABASE_CODES_DESC = {
    'WIKI': 'EOD stock prices of 3000 US companies',
    'CME': 'Chicago Mercantile Exchange futures data',

}
QUANDL_DATABASE_CODES = ['WIKI', 'CME']
QUANDL_API_KEY = 'Qzp4XmsULyjBPhy3j7wg'
QUANDL_URL = 'www.quandl.com'
QUANDL_URL2 = '/api/v3/datasets/{database_code}/{dataset_code}.csv/'
LIMIT_MAX = 999999999

#######################################################################################################################
# ALPHA VANTAGE - stock prices, stock vol, technical indicators
#######################################################################################################################
AV_API = 'AlphaVantage API'
ALPHA_VANTAGE_API_KEY = 'VMYCYOBENJ9KCIVI'
# Data is returned in CSV or JSON format
AV_STOCK_API_FUNCTIONS = {'TIME_SERIES_INTRADAY': ['function', 'symbol', 'interval', 'datatype'],
                          'TIME_SERIES_DAILY': ['function', 'symbol', 'datatype'],
                          'TIME_SERIES_DAILY_ADJUSTED': ['function', 'symbol', 'datatype'],
                          'TIME_SERIES_WEEKLY': ['function', 'symbol', 'datatype'],
                          'TIME_SERIES_WEEKLY_ADJUSTED': ['function', 'symbol', 'datatype'],
                          'TIME_SERIES_MONTHLY': ['function', 'symbol', 'datatype'],
                          'TIME_SERIES_MONTHLY_ADJUSTED': ['function', 'symbol', 'datatype'],
                          'BATCH_STOCK_QUOTES': ['function', 'symbols', 'datatype']}

# Data is returned in JSON format only.
# For more info visit https://www.alphavantage.co/documentation/#technical-indicators
AV_TECHINDI_API_FUNCTIONS = {'SMA': ['function', 'symbol', 'interval', 'time_period', 'series_type'],
                             'EMA': ['function', 'symbol', 'interval', 'time_period', 'series_type'],
                             'WMA': ['function', 'symbol', 'interval', 'time_period', 'series_type'],
                             'DEMA': ['function', 'symbol', 'interval', 'time_period', 'series_type'],
                             'TEMA': ['function', 'symbol', 'interval', 'time_period', 'series_type'],
                             'TRIMA': ['function', 'symbol', 'interval', 'time_period', 'series_type'],
                             'KAMA': ['function', 'symbol', 'interval', 'time_period', 'series_type'],
                             'MAMA': ['function', 'symbol', 'interval', 'series_type', 'fastlimit','slowlimit'],
                             'T3': ['function', 'symbol', 'interval', 'time_period', 'series_type'],
                             'MACD': ['function', 'symbol', 'interval', 'series_type'],
                             'MACDEXT': ['function', 'symbol', 'interval', 'series_type'],
                             'STOCH': ['function', 'symbol', 'interval'],
                             'RSI': ['function', 'symbol', 'interval', 'time_period', 'series_type'],
                             'ADX': ['function', 'symbol', 'interval', 'time_period', 'series_type'],
                             'BBANDS': ['function', 'symbol', 'interval', 'time_period', 'series_type'],
                             }
AV_URL = 'www.alphavantage.co'
AV_URL2 = '/query/'
