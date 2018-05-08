"""
This module contains the parent class for all core financial data APIs
"""
import enum
from common.consts import *
from abc import ABCMeta, abstractmethod


class FinanceDataType(enum.Enum):
    STOCK = 1
    FX = 2
    FUTURES = 3
    CRYPTOCURRENCY = 4
    INDEX = 5
    TECH_INDICATOR = 5
    SENTIMENT = 6
    OTHER = 7


class FinanceData(object):
    """Abstract base class for all Finance Data API classes"""

    __metaclass__ = ABCMeta

    def __init__(self, finance_data_type):
        self._finance_data_type = finance_data_type

    @property
    def dtype(self):
        return self._finance_data_type

    @abstractmethod
    def download_data(self, *args, **kwargs):
        """Abstract method for downloading any finance data"""
        pass


def validate_params(func):
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        fdatatype = args[0]._finance_data_type
        api = args[0].API

        if api == AV_API:
            if fdatatype == FinanceDataType.STOCK:
                function = kwargs.get('function')
                assert function and function in AV_STOCK_API_FUNCTIONS.keys()
                assert all(param in kwargs for param in AV_STOCK_API_FUNCTIONS[function])
            elif fdatatype == FinanceDataType.TECH_INDICATOR:
                function = kwargs.get('function')
                assert function and function in AV_TECHINDI_API_FUNCTIONS.keys()
                assert all(param in kwargs for param in AV_TECHINDI_API_FUNCTIONS[function])
            elif fdatatype == FinanceDataType.FX:
                pass
            elif fdatatype == FinanceDataType.FUTURES:
                pass
            elif fdatatype == FinanceDataType.CRYPTOCURRENCY:
                pass
            elif fdatatype == FinanceDataType.SENTIMENT:
                pass
            elif fdatatype == FinanceDataType.CRYPTOCURRENCY:
                pass

        elif api == Q_API:
            if func_name == 'download_csv':
                if fdatatype == FinanceDataType.STOCK:
                    start_date = kwargs.get('start_date')
                    end_date = kwargs.get('end_date')
                    dataset_code = kwargs.get('dataset_code')
                    database_code = kwargs.get('database_code')
                    assert start_date
                    assert end_date
                    assert dataset_code
                    assert database_code in QUANDL_DATABASE_CODES
                elif fdatatype == FinanceDataType.FX:
                    pass
                elif fdatatype == FinanceDataType.FUTURES:
                    pass
                elif fdatatype == FinanceDataType.CRYPTOCURRENCY:
                    pass
                elif fdatatype == FinanceDataType.SENTIMENT:
                    pass
                elif fdatatype == FinanceDataType.CRYPTOCURRENCY:
                    pass

            elif func_name == 'download_data':
                if fdatatype == FinanceDataType.STOCK:
                    database_code = kwargs.get('database_code')
                    dataset_code = kwargs.get('dataset_code')
                    assert database_code
                    assert dataset_code
                    assert isinstance(dataset_code, str)
                    assert isinstance(database_code, str)
                elif fdatatype == FinanceDataType.FX:
                    pass
                elif fdatatype == FinanceDataType.FUTURES:
                    pass
                elif fdatatype == FinanceDataType.CRYPTOCURRENCY:
                    pass
                elif fdatatype == FinanceDataType.SENTIMENT:
                    pass
                elif fdatatype == FinanceDataType.CRYPTOCURRENCY:
                    pass
                elif fdatatype == FinanceDataType.OTHER:
                    database_code = kwargs.get('database_code')
                    dataset_code = kwargs.get('dataset_code')
                    assert database_code
                    assert dataset_code
                    assert isinstance(dataset_code, str)
                    assert isinstance(database_code, str)

        return func(*args, **kwargs)

    return wrapper
