"""
Using scikitlearn to predict the stock price

"""
# Imports
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from Prepare_Data import Data
import copy


# User Prepare Data to get csv


class Stock_Classifier:

    def __init__(self, stock_name: str):
        self.dataset = pd.read_csv(stock_name)
        self.dataset = self.filter_nan(self.dataset)
        self.Y = self.dataset['Close']
        self.dataset.drop('Close', inplace=True, axis=1)
        self.X = self.dataset

    def filter_nan(self, dataset: pd.DataFrame):
        """
        This methods filters out NaN value if there is a NaN in that Column
        :param dataset: DataFrame
        :return: dataset
        """
        features = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
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
        for k in range(len(new_Y)):
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

        :param X: DataFrame
        :param Y: DataFrame -> should be convert to numeric in 1 0
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


