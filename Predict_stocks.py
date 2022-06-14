"""
Using scikitlearn to predict the stock price

"""
# Imports
import random

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from Prepare_Data import Data
import copy
from stockstats import StockDataFrame
from sklearn.neural_network import MLPClassifier


## Use keras on Desktop
# from keras.models import Sequential
# from keras.layers.core import Dense


# User Prepare Data to get csv


class Stock_Classifier:

    def __init__(self, stock_name: str, test=False):
        self.dataset = pd.read_csv(stock_name)
        self.dataset = self.filter_nan(self.dataset, ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'])
        if test:
            self.append_new_row()
            # print(self.dataset)
        ## Separate dataset into Stockstats
        self.stockstat = self.data_to_stockstats(self.dataset)
        self.Y = self.dataset['Close']
        self.Y = self.Y.iloc[1:]
        self.dataset.drop('Close', inplace=True, axis=1)
        self.X = self.dataset
        self.add_technial_indicators()
        self.X = self.filter_nan(self.X, ['RSI', 'SMA', 'ST'])
        # print(self.X)

    def data_to_stockstats(self, df):
        stock = StockDataFrame.retype(df[["Open", "Close", "High", "Volume", "Low"]])
        return stock

    def append_new_row(self):
        close_price = self.dataset.iloc[-1]['Close']
        high = close_price + random.choice([5, 6, 7, 8, 9, 10])
        low = close_price - random.choice([5, 6, 7, 8, 9, 10])
        adj_close = close_price + 5
        volume = self.dataset.iloc[-1]['Volume'] - 1000000
        new_data = {'Open': close_price, 'High': high, 'Low': low, 'Close': close_price - 10, 'Adj Close': adj_close,
                    'Volume': volume}
        # print(new_data)
        self.dataset = self.dataset.append(new_data, ignore_index=True)

    def add_technial_indicators(self):
        self.X.insert(loc=1, column='RSI', value=self.stockstat["rsi"])
        self.X.insert(loc=2, column='SMA', value=self.stockstat["open_10_sma"])
        self.X.insert(loc=3, column='ST', value=self.stockstat["supertrend"])
        print("Added Technical indicators")

    def filter_nan(self, dataset: pd.DataFrame, features):
        """
        This methods filters out NaN value if there is a NaN in that Column
        :param dataset: DataFrame
        :param features: List of Features
        :return: dataset
        """
        for i in features:
            dataset.dropna(subset=[i], inplace=True)

        return dataset

    def filter_in_numeric(self):
        """
        Use this only if you want to filter the predict value in a numeric 0 and 1
        if Close is higher than Open this means the stock would be a buy on that day else
        a sell. 1 Buy 0 Sell
        :return: dataset
        """
        new_Y = copy.deepcopy(self.Y)
        open_X = self.X['Open']
        for k in range(1, len(new_Y) + 1):
            if new_Y[k] > open_X[k]:
                new_Y[k] = 1
            else:
                new_Y[k] = 0
        return new_Y

    def linear_Regression(self):
        """
        Predicting the Value using linear Regression
        :param XTrain:
        :param YTrain:
        :param XTest:
        :param YTest:
        :return: None
        """
        XTrain, XTest, YTrain, YTest = train_test_split(self.X, self.Y.astype(int), test_size=0.3)
        l_clf = LinearRegression()
        l_clf = l_clf.fit(XTrain, YTrain)
        Y_Predict = l_clf.predict(XTest)
        # print(Y_Predict)
        Y_Predict = np.rint(Y_Predict)
        # print(Y_Predict)
        YY = YTest.to_numpy()
        YY = np.rint(YY)
        # print(yy)
        correct = 0
        incorrect = 0
        for i in range(YY.shape[0]):
            if Y_Predict[i] == YY[i]:
                correct += 1
            else:
                incorrect += 1

        print("From %d test cases, %d classified as correct and %d false " % (YY.shape[0], correct, incorrect))
        print("\n  Accuracy score : ", (correct / YY.shape[0]))

    def random_forest_classifier(self):
        """
        :return: None
        """
        new_Y = self.filter_in_numeric()
        XTrain, XTest, YTrain, YTest = train_test_split(self.X, new_Y.astype(int), test_size=0.3)
        r_clf = RandomForestClassifier(max_depth=10)
        r_clf = r_clf.fit(XTrain, YTrain)
        predicted = r_clf.predict(XTest)
        print("Result for Ytest : ", YTest.to_numpy(), "\n")
        print("Result for Prediction : ", predicted, "\n")
        print("Accuracy score: ", accuracy_score(YTest, predicted))

    def MLPClassifier(self, test_data=None, y_test=None):

        new_Y = self.filter_in_numeric()
        # print(new_Y)

        XTrain, XTest, YTrain, YTest = train_test_split(self.X, new_Y.astype(int), test_size=0.3)
        # print(YTrain)
        sc = StandardScaler()
        xtrain_scaled = sc.fit_transform(XTrain)
        if test_data is not None:
            xtest_scaled = sc.fit_transform(test_data)
        else:
            xtest_scaled = sc.fit_transform(XTest)
        clf = MLPClassifier(hidden_layer_sizes=(50, 50, 60, 30, 9), activation='relu', solver='adam').fit(xtrain_scaled,
                                                                                                          YTrain)
        predicted = clf.predict(xtest_scaled)
        print(predicted)
        if y_test is not None:
            print(y_test.to_numpy())
            print("Accuracy score : ", accuracy_score(y_test, predicted))
        else:
            print(YTest.to_numpy())
            print("Accuracy score : ", accuracy_score(YTest, predicted))


st = Stock_Classifier('TSLA')
st2 = Stock_Classifier('TSLA-Test2', test=True)
st2.Y = st2.filter_in_numeric()
st.MLPClassifier(test_data=st2.X, y_test=st2.Y)
