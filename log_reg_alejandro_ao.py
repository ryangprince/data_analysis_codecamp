# PRACTICE PROJECT

# * Logistic Regression Project: Cancer Prediction with Python
# * Link to YouTube: https://youtu.be/My4JgIeFdWk?si=FOA8iMtV99Zo3CwR
# * Link to kaggle data set: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data


import pandas as pd
import seaborn as sns


data = pd.read_csv("data/raw/breast_cancer.csv")
data.head()
data.info()
data.describe()
data.shape


# * Clean the data

sns.heatmap(data.isnull())

data.drop(["id", "Unnamed: 32"], axis=1, inplace=True)
data.head()


# * Convert 'diagnosis' column into ones and zeros so the model can use it

"""
How update was made in video.

data["diagnosis"] = [1 if value == "M" else 0 for value in data["diagnosis"]]
data.head()
data["diagnosis"].value_counts()
"""

# * alternate method using replace()

data = data.replace({"diagnosis": {"M": 1, "B": 0}})
data.head()
data["diagnosis"].value_counts()


# * Change 'diagnosis' dtypes from int to category

data["diagnosis"] = data["diagnosis"].astype("category", copy=False)
data["diagnosis"].value_counts().plot(kind="bar")


# * Divide into target variables and predictors

y = data["diagnosis"]
X = data.drop(["diagnosis"], axis=1)
y
X


# * Normalize the data

from sklearn.preprocessing import StandardScaler

# * Create scaler object
scaler = StandardScaler()

# * fit the scaler to the data and transform it
X_scaled = scaler.fit_transform(X)
X_scaled


# * Split the data into Test and Train sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.30, random_state=42
)


# * train the model

from sklearn.linear_model import LogisticRegression

# * create the logistic regression (lr) model
lr = LogisticRegression()

lr.fit(X_train, y_train)

# * predict the target variable based on test data
y_pred = lr.predict(
    X_test,
)
y_pred


# * evaluation of the model

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy}:.2f")

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
