# PRACTICE PROJECT

# * Exploratory Data Analysis with Pandas Python
# * Link to YouTube: https://youtu.be/xi0vhXFPegw?si=3kcGYvr7kw7xBpBF
# * Link to kaggle notebook

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("ggplot")
# pd.set_option('max_columns', 200)

df = pd.read_csv("data/raw/coaster_db.csv")


# * BASIC DATA UNDERSTANDING

df.shape

df.head()

df.columns

df.dtypes

df.describe()

# * Remove unnecessary columns

df = df[
    [
        "coaster_name",
        "Location",
        "Status",
        "Manufacturer",
        "year_introduced",
        "latitude",
        "longitude",
        "Type_Main",
        "opening_date_clean",
        "speed_mph",
        "height_ft",
        "Inversions_clean",
        "Gforce_clean",
    ]
].copy()

df.shape
df.head()

# * Identify different dtypes in 'opening_date_clean' column
# ! UNUSED, changed dates in csv for time
# def mixed_dtypes_datetime(df, column_name):

#     column_dtype = pd.api.types.infer_dtype(df[column_name])
#     if column_dtype == 'mixed':
#         print("The column contains mixed data types.")

# mixed_dtypes_datetime(df, 'opening_date_clean')

# def unique_dtypes_datetime(df, column_name):

#     unique_types = df[column_name].apply(type).unique()
#     print("Unique data types in the column:")

#     ls_unique_types = []

#     for dtype in unique_types:
#         # print(dtype)
#         ls_unique_types.append(dtype)

#     return ls_unique_types

# unique_dtypes_datetime(df, 'opening_date_clean')


# * Convert 'opening_date_clean' to datetime.

df["opening_date_clean"] = pd.to_datetime(df["opening_date_clean"])

df["opening_date_clean"].head(15)


# * Change column names

df.columns
df = df.rename(
    columns={
        "coaster_name": "Coaster_Name",
        "year_introduced": "Year_Introduced",
        "opening_date_clean": "Opening_Date",
        "speed_mph": "Speed_mph",
        "height_ft": "Height_ft",
        "Inversions_clean": "Inversions",
        "Gforce_clean": "Gforce",
    }
)

df.head()


# * Identify missing values in the data frame and how often they occur

df.isna().sum()

df.loc[df.duplicated()]

df.loc[df.duplicated(subset=["Coaster_Name"])].head(5)

# * Checking an example duplicate

df.query('Coaster_Name == "Crystal Beach Cyclone"')

# * Find columns where name, location, and opening date are duplicated
# * and find the total number of rows that are duplicates

df.duplicated(subset=["Coaster_Name", "Location", "Opening_Date"]).sum()

# * Find the columns that are not duplicates
~df.duplicated(subset=["Coaster_Name", "Location", "Opening_Date"])

"""
Save the columns that are not duplicates so the duplicates. are no longer in 
the data frame.
- This will cause a problem with the indexes which will not change. 
- To fix this we set the parameter `drop=True` and that will drop the original index.
"""
df = df.loc[
    ~df.duplicated(subset=("Coaster_Name", "Location", "Opening_Date"))
].reset_index(drop=True)

df.head()


# * FEATURE UNDERSTANDING (univariate analysis)

ax = (
    df["Year_Introduced"]
    .value_counts()
    .head(10)
    .plot(kind="bar", title="Top 10 Years Coasters Introduced")
)
ax.set_xlabel("Year Introduced")
ax.set_ylabel("Count")

ax1 = df["Speed_mph"].plot(kind="hist", bins=20, title="Coaster Speed (mph)")
ax1.set_xlabel("Speed (mph)")

ax2 = df["Speed_mph"].plot(kind="kde", title="Coaster Speed (mph)")
ax2.set_xlabel("Speed (mph)")


# * FEATURE RELATIONSHIPS

df.plot(kind="scatter", x="Speed_mph", y="Height_ft")
plt.show()


ax3 = sns.scatterplot(x="Speed_mph", y="Height_ft", hue="Year_Introduced", data=df)
ax3.set_title("Coaster Speed vs. Height")
plt.show()


sns.pairplot(
    df,
    vars=["Year_Introduced", "Speed_mph", "Height_ft", "Inversions", "Gforce"],
    hue="Type_Main",
)
plt.show()


df_corr = (
    df[["Year_Introduced", "Speed_mph", "Height_ft", "Inversions", "Gforce"]]
    .dropna()
    .corr()
)

sns.heatmap(df_corr, annot=True)


# * ASK A QUESTION ABOUT THE DATA

# * What are the locations with the fastest roller coasters (minimum of 10)?

df["Location"].value_counts()

ax4 = (
    df.query('Location != "Other"')
    .groupby("Location")["Speed_mph"]
    .agg(["mean", "count"])
    .query("count >= 10")
    .sort_values("mean")["mean"]
    .plot(kind="barh", figsize=(12, 5), title="Average Coaster Speed by Location")
)
ax4.set_xlabel("Average Coaster Speed")
plt.show()
