
# Prediction Stocks Project
---
This is only a private project. The prediction is not considered to be used to buy a stock.
This is no financial advice. Use this at your own risk.
<br>
All data belongs to yfinance.

---

## Introduction
---

This is a project to use Machine Learning Models from scikit-learn to predict a stock price.
The stock market is very risky to invest in if you don't know how to. Therefore, this project should help 
to predict a stock price. However, the prediction can not be 100%, since a stock moves depending on good news or bad news.
In this project we will look into different ML Models, wheter or not the ML did a good prediction.

---

### Prepare Data Class
---
Prepare Data is a class where we get a the csv data so later on we can filter out unnecessary Columns and use it 
to predict a stock. 

#### How it works:
```python
from Prepare_Data import Data
# Create a new instance from data
# Data uses a stock name to fetch data from yfinance
# Set the interval yourself
# How yfinance works https://github.com/ranaroussi/yfinance
data = Data('MSFT', start='2021-01-01', end='2022-01-01', interval='1d')
# Note that load_data has a params case, which Rename your csv default is ""
data.load_data()
```
#### Note that load_data will fetch the data from yfinance and turn the table into csv file

---


### Predict_stocks Class
---
Predict_stocks class reads the csv file and filter out the NaN rows. We then prepare the data for X and Y.
Since many things can influence the stock price we only use the column from the yfinance.
We use all the columns except Close. Close should be our prediction, it tells us if in that day was a buy or not.

#### How it works:
```python
from Predict_stocks import Stock_Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# Read the stock file
stock = Stock_Classifier('MSFT')
#Using linear Regression
stock.linear_Regression()
#It should print out the accuracy score
# Using Random Forest Classifier
stock.random_forest_classifier()

```
Note that in Random Forest Classifier filter out Close and turn it into 0 and 1.
We turn it into 1 if Close was higher than Open, which means on that day the stock was a buy and otherwise a sell.


---

### Example

This is an example for the AAPL stock. We get the following result for Random_Forest

```
Result for Ytest :  [0 1 1 1 0 0 0 0 1 0 0 0 1 0 1 0 1 0 1 0 1 1 0 0 1 1 1 1 1 1 1 0 0 1 1 0 0
 1 0 1 1 0 1 0 1 1 0 0 1 1 1 1 0 1 0 0 1 1 0 1 1 0 1 1 0 1 0 1 0 0 1 1 1 1
 0 1 0 0 1 0 1 1 0 0 0 0 0 1 0 1 0 1 0 0 0 1 1 0 0 1 0 0 1 1 1 1 0 1 0 0 0
 1 0 1 1 0 1 0 0 0 1 0 1 0 0 1 0 0 0 0 1 0 1 1 1 1 0 1 0 1 1 1 1 0 0 0 0 1
 0 0 1 1 1 1 0 1 1 1 1 1 1 0 0 1 0 0 1 1 1 1 1 1 1 1 0 0 1 1 1 0 0 0 0 1 0
 0 0 1 1 1 0 0 1 0 1 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 1 0 0 1 1 0 1 0 1] 

Result for Prediction :  [0 1 1 0 0 0 0 0 1 0 0 1 0 0 0 0 1 0 1 0 1 1 1 0 1 1 1 1 1 1 1 0 0 1 1 0 0
 1 0 1 0 0 1 0 1 0 0 0 0 1 1 1 0 1 0 0 1 1 0 0 0 0 1 1 0 0 0 0 0 0 1 0 1 1
 0 1 0 0 1 0 1 1 0 0 0 0 0 1 0 1 0 1 0 0 0 1 1 0 0 1 0 0 0 1 1 1 0 1 0 0 1
 1 1 1 1 1 0 0 0 1 1 0 1 0 0 1 0 1 0 0 1 0 1 1 1 1 0 1 0 1 1 1 1 0 0 1 0 1
 0 0 0 1 1 0 0 1 1 1 1 1 1 1 0 1 0 0 1 1 1 1 1 0 0 1 0 0 0 1 1 0 1 0 0 1 0
 0 0 1 0 1 0 0 1 0 1 0 1 0 1 0 0 0 0 0 1 0 0 0 0 0 1 0 1 1 1 0 1 0 1] 

Accuracy score:  0.8493150684931506

```

Another Example with Linear Regression

```
From 219 test cases, 216 classified as correct and 3 false 

  Accuracy score :  0.9863013698630136

```

Note that the accuracy score is very high, but the price movement on the stock can not really be predicted with such an ML
model. Still it can be used to predict a price movement.