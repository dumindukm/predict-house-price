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
%matplotlib inline


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

#%%

#Let's visualize data
print(data.describe())

#%%
#Let's prepare data from model training
Y = data['PRICE']
X = data.drop('PRICE', axis = 1)


#%%
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.30, random_state = 5)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

#%%
lm = LinearRegression()
lm.fit(X_train, Y_train)

Y_pred = lm.predict(X_test)

plt.scatter(Y_test, Y_pred)
plt.xlabel("Prices")
plt.ylabel("Predicted prices")
plt.title("Prices vs Predicted prices")

#%%
mse = sklearn.metrics.mean_squared_error(Y_test, Y_pred)
print(mse)

#%%

r2_score(Y_test, Y_pred)

#%%

index = 0
results = lm.predict([X_test.loc[index]])
print("predicted price",results[0])

print("Actual price ",Y_test.loc[index])
#%%
X_test.keys

#%%
