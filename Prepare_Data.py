"""
Preparing Data from Yahoo Finance API
Upon on that separate all the data so we can predict the stock price

Notes:
    All data belongs to Yahoo Finance

"""
# Imports
import numpy as np
import yfinance as yf
import pandas as pd


class Data:

    def __init__(self, stocks_name: str, start: str, end: str, interval: str):
        """
        Constructor params
        :param stocks_name: String
        :param start: String
        :param end: String
        :param interval: String
        """
        self.stocks_name = stocks_name
        self.data = None
        self.start = start
        self.end = end
        self.interval = interval

    def load_data(self, case=""):
        """
        Fetch Data from Yfinance
        :return: DataFrame from yf
        """
        self.data = yf.download(self.stocks_name, start=self.start, end=self.end,interval=self.interval)
        self.data.to_csv(self.stocks_name + case, index=False)
        return self.data



    def __str__(self):
        """
        represents the class
        :return: DataFrame.to_string : String
        """
        return self.data.to_string()

