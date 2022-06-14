
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
### MLP Classifier 

Added a new Classifier from scikitlearn, which trains data with a Neural Network.
For the input layer we are using our features and the new features, which are: RSI,
SMA and Supertrend. These features are used to give more information to the neural network
for our input layer.

<br/>
The technical indicators are added automatically so, the training data gets more precisely.


####Example

```python
#We also scaled the data with StandardScaler
xtrain_scaled = sc.fit_transform(XTrain)
if test_data is not None:
    xtest_scaled = sc.fit_transform(test_data)
else:
    xtest_scaled = sc.fit_transform(XTest)
#We need to scale the price so it gets more precisely
#Using the MLP classifier with hiddens layers
clf = MLPClassifier(hidden_layer_sizes=(50, 50, 60, 30, 9), activation='relu', solver='adam').fit(xtrain_scaled,
                                                                                                          YTrain)
```
#### How it works
Note that our MLPClassifier takes a test_data parameter (default is None) and y_test (default is also None)
```python
st = Stock_Classifier('TSLA')
st.MLPClassifier()
#Trains the Data and Predict the Price
```
---

### New Feature

I added a new feature which could be call recursively. This feature append a new row to the csv data.
We take the Close Price and added this to a new row as the Open Price - Note that this is not very precise since there is After Hour
and Pre Market -. Therefore, we could use this data to predict the future trend, wheter the stock is going up or down.
To append a new row use:
```python
#The Stock_Classifiert takes a param test, which append a new row to the csv data
st2 = Stock_Classifier('TSLA-Test2', test=True)
st2.Y = st2.filter_in_numeric()
st.MLPClassifier(test_data=st2.X, y_test=st2.Y)
# We can use this to predict the stock data from st2
# Training the model and test to a new test data

```


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