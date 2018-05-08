from marketdata.quandlapi import *
from marketdata.alphavantageapi import *
from marketdata.financedata import *

obj = AlphaVantageAPI(FinanceDataType.STOCK)
# print obj.dtype
json = obj.download_data(function='TIME_SERIES_INTRADAY', symbol='AAPL', interval='60min', outputsize='compact',
                         datatype='json')[1]
AlphaVantageAPI.json_to_df(json)
#print obj.download_data(function='SMA', symbol='MSFT', interval='15min', outputsize='compact',
#                       datatype='csv', time_period='60', series_type='close')[1]
# start_date = "2012-10-15"
# end_date = "2016-01-06"
# obj = QuandlAPI(FinanceDataType.STOCK)
# df = obj.download_data(database_code='WIKI',dataset_code='AAPL',start_date=start_date, end_date=end_date)[1]
# df.to_csv('/Users/shivankurkapoor/Desktop/Study/DeepStocks/Data/stockdata_aapl.csv')

#print obj.download_data(database_code='CBOE',dataset_code='VIX',start_date=start_date, end_date=end_date)[1]


import pandas
obj = AlphaVantageAPI(FinanceDataType.TECH_INDICATOR)
SMA = obj.download_data(function='SMA', symbol='AAPL', interval='daily', time_period=60, series_type='close')[1]
EMA = obj.download_data(function='EMA', symbol='AAPL', interval='daily', time_period=60, series_type='close')[1]
MACD = obj.download_data(function='MACD', symbol='AAPL', interval='daily', time_period=60, series_type='close')[1]
STOCH = obj.download_data(function='STOCH', symbol='AAPL', interval='daily', time_period=60, series_type='close')[1]
RSI = obj.download_data(function='RSI', symbol='AAPL', interval='daily', time_period=60, series_type='close')[1]
ADX = obj.download_data(function='ADX', symbol='AAPL', interval='daily', time_period=60, series_type='close')[1]
BBANDS = obj.download_data(function='BBANDS', symbol='AAPL', interval='daily', time_period=60, series_type='close')[1]

indicators = {
    'SMA' : SMA,
    'EMA' : EMA,
    'MACD' : MACD,
    'STOCH' : STOCH,
    'RSI' : RSI,
    'ADX' : ADX,
    'BBANDS' : BBANDS
}

for k,v in indicators.items():
    df = AlphaVantageAPI.json_to_df(v)
    df.to_csv('/Users/shivankurkapoor/Desktop/Study/DeepStocks/Data/' + k + '.csv')