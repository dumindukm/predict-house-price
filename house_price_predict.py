#%%

# Let's import some packages
import numpy as np 
import pandas as pd
from sklearn.model_selection import ShuffleSplit
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import r2_score
# special matplotlib argument for improved plots
from matplotlib import rcParams

# Pretty display for notebooks
#%matplotlib inline


#%%
# Let's load data

boston = load_boston()

#%%

# Display what is in dataste
print(boston.keys())

#%%
# Display column names
print(boston.DESCR)

#%%
data = pd.DataFrame(boston.data)
data.columns = boston.feature_names
data['PRICE'] = boston.target

print(data.head())

corr = data.corr()

print("Co-relation to price",corr['PRICE'].sort_values(ascending=False))
#%%

#Let's visualize data
print(data.describe())

#%%
#Let's prepare data from model training
Y = data['PRICE']
X = data.drop('PRICE', axis = 1)

# With the help of co-relation select only few columns for train
min_features = ['RM','ZN','B','DIS','CHAS','AGE','CRIM','INDUS','PTRATIO','TAX']
X_minimum_columns = X[min_features]

#%%
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.30, random_state = 5)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

# Method for Train model using Linear Regression
def TrainUsingLineearRegression(x,y):
    lm = LinearRegression()
    lm.fit(x, y)
    return lm

def PlotResultsUsingScattePlot(y_test, y_pred):
    plt.scatter(y_test, y_pred)
    plt.xlabel("Prices")
    plt.ylabel("Predicted prices")
    plt.title("Prices vs Predicted prices (Linear regression)")

lm = TrainUsingLineearRegression(X_train, Y_train)
# do prediction using Linear regression
Y_pred = lm.predict(X_test)

PlotResultsUsingScattePlot(Y_test, Y_pred)

#%%
def EvalateModel(y_test, y_pred):
    mse = sklearn.metrics.mean_squared_error(y_test, y_pred)
    print("Mean squarred error for full features",mse)
    print("R2 score for full features",r2_score(y_test, y_pred))

def DoPredictFromTestset(itemIndex, model, x_test, y_test):
    results = model.predict([x_test.iloc[itemIndex]])
    print("predicted price",results[0])
    print("Actual price ",y_test.iloc[itemIndex])

print('***** predict results using full features ******')

EvalateModel(Y_test, Y_pred)
DoPredictFromTestset(1,lm,X_test, Y_test)
DoPredictFromTestset(2,lm,X_test, Y_test)
DoPredictFromTestset(3,lm,X_test, Y_test)

#%%
print("Start training using few models")
#Let's train model using few features
# Let's train model using fewer features
lm = TrainUsingLineearRegression(X_train[min_features], Y_train)
# do prediction using Linear regression
Y_pred = lm.predict(X_test[min_features])

PlotResultsUsingScattePlot(Y_test, Y_pred)
print('***** predict results using few features ******')

EvalateModel(Y_test, Y_pred)
DoPredictFromTestset(1,lm,X_test[min_features], Y_test)
DoPredictFromTestset(2,lm,X_test[min_features], Y_test)
DoPredictFromTestset(3,lm,X_test[min_features], Y_test)
#%%

# Let's predict using DeciosnTree
print("****** Start train using DecisionTreeRegressor ******")
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)

regressor.fit(X_train, Y_train)

Y_pred = regressor.predict(X_test)
EvalateModel(Y_test, Y_pred)
DoPredictFromTestset(1,regressor,X_test, Y_test)
DoPredictFromTestset(2,regressor,X_test, Y_test)
DoPredictFromTestset(3,regressor,X_test, Y_test)
#%%


#%%
