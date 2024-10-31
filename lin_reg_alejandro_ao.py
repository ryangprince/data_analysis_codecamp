# PRACTICE PROJECT

# * Linear Regression in Python - Full Project for Beginners
# * Link to YouTube: https://youtu.be/O2Cw82YR5Bo?si=A1ZQKvf8KgyR9uZR
# * Link to kaggle data set: https://www.kaggle.com/datasets/kolawale/focusing-on-mobile-app-or-website

import pandas as pd
import pylab
import math
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

df = pd.read_csv("data/raw/Ecommerce_Customers.csv")

df.head()

df.info()

df.describe()

# * EDA

sns.jointplot(x="Time on Website", y="Yearly Amount Spent", data=df, alpha=0.5)

sns.jointplot(x="Time on App", y="Yearly Amount Spent", data=df, alpha=0.5)

sns.pairplot(df, kind="scatter", plot_kws={"alpha": 0.4})


sns.lmplot(
    x="Length of Membership",
    y="Yearly Amount Spent",
    data=df,
    scatter_kws={"alpha": 0.3},
)


# * Split the data into the training set and targets

X = df[
    ["Avg. Session Length", "Time on App", "Time on Website", "Length of Membership"]
]
X.head()

y = df["Yearly Amount Spent"]
y.head()

"""
train_test_split()
- train_size :  select the percentage of the data set you want to train. 
                If the input is 0.3 then 30% of the data will be used for testing
                and 70% for training.
-random_state : Controls the shuffling applied to the data before applying the split. 
                Pass an int for reproducible output across multiple function calls.    
"""

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

X_train.shape
X_test.shape
y_train.shape
y_test.shape


# * Training the model

lm = LinearRegression()

lm.fit(X_train, y_train)

lm.coef_

cdf = pd.DataFrame(lm.coef_, X.columns, columns=["Coef"])
print(cdf)


# * Predictions

predictions = lm.predict(X_test)
predictions


# * Plot the predictions

sns.scatterplot(x=predictions, y=y_test)
plt.xlabel("Predictions")
plt.title("Evaluation of our ML model")


print("Mean Absolute Error: ", mean_absolute_error(y_test, predictions))
print("Mean Absolute Error: ", mean_squared_error(y_test, predictions))
print("RMSE: ", math.sqrt(mean_squared_error(y_test, predictions)))


# * Residuals -> actual values minus predictions, should fall in a normal distribution

residuals = y_test - predictions
residuals

sns.displot(residuals, bins=30, kde=True)

stats.probplot(residuals, dist="norm", plot=pylab)
pylab.show()
