# PRACTICE PROJECT

# * Python for Data Science Course – Hands-on Projects with EDA, AB Testing & Business Intelligence
# * Link to YouTube: https://youtu.be/FTpmwX94_Yo?si=vw-3m3265jBP2ESB
# * Link to 'percent_bachelors_women_usa.csv': https://github.com/johnashu/datacamp/blob/master/percent-bachelors-degrees-women-usa.csv
# * Link to GitHub repo for Project 1: https://github.com/TatevKaren/CaseStudies/tree/main/AB%20Testing
# * Link to GitHub repo for Project 2: https://github.com/TatevKaren/CaseStudies/tree/main/SuperStoreCaseStudy

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


data_csv = pd.read_csv("data/raw/percent-bachelors-degrees-women-usa.csv")

# * Part 1: Data Exploration and Preprocessing

data_csv.head()
data_csv.shape
data_csv.columns
data_csv.info()

"""
data_csv.dropna() -> drops all null values
data_csv.fillna('') -> replaces all null values
"""

# * For each column get the sum of all the duplicate values
data_csv.loc[data_csv.duplicated()].sum()
data_csv.drop_duplicates()

data_csv.iloc[10]
data_csv.loc[data_csv["Year"] == 1980]


# * Part 1: Filtering, Sorting, Grouping

data = pd.DataFrame(
    {
        "Name": ["Anna", "Karen", "John", "Alice", "Kevin", "Sanna", "Bob", "Emily"],
        "Age": [35, 30, 57, 65, 25, 19, 20, 65],
        "Salary": [20000, 60000, 145000, 170000, 30000, 10000, 220000, 120000],
        "Department": [
            "Tech",
            "Tech",
            "Tech",
            "Healthcare",
            "Operations",
            "Operations",
            "Tech",
            "Tech",
        ],
    }
)

data.sort_values(by="Salary", ascending=True)

data.sort_values(by="Salary", ascending=False)


data.groupby("Department").count()

data.groupby("Department")["Name"].count()

data.groupby("Department")["Salary"].mean()

data[data["Salary"] > 100000]

(data[(data["Salary"] > 100000) & (data["Salary"] < 200000)])


# * Part 1: Descriptive Statistics


desc_stats = [100, 20, 5, 20, 45, -100, 46]

# * if the mean is close to the median it means we're most likely NOT working with a skewed distribution
np.mean(desc_stats)
np.median(desc_stats)

stats.mode(desc_stats)

"""
VARIANCE measures the spreadness or the dispersion of the data.
- It quantifies how far your number is from the mean.
- A higher variance indicates a greater variability in your data.
- A lower variance indicates a smaller variability in your data.

STANDARD DEVIATION is the square root of the variant.
- Provides a standardized way to explain the average distance between each data point and the mean.
"""

np.var(desc_stats)
np.std(desc_stats)

data_csv.describe()


# * Part 1: Merging & Joins

merg1 = pd.DataFrame(
    {"key": ["A", "B", "C", "D", "E", "F", "G"], "value1": [1, 2, 3, 4, 5, 6, 7]}
)

merg2 = pd.DataFrame(
    {"key": ["C", "D", "E", "F", "G", "H"], "value2": [8, 9, 10, 11, 12, 13]}
)

"""
Inner join
expected output: those keys that appear in merg1 and merg2
along with their expected values will appear in the new data frame.
"""

merge_inner_join = pd.merge(merg1, merg2, on="key", how="inner")
merge_inner_join

"""
Left join
expected output: all the values that appear in the first data frame (left table),
as well as any values from the second data frame (right table) that are also in the
left table.
"""

merge_left_join = pd.merge(merg1, merg2, on="key", how="left")
merge_left_join

"""
Right join
expected output: all the values that appear in the second data frame (right table),
as well as any values from the first data frame (left table) that are also in the
right table.
"""

merge_right_join = pd.merge(merg1, merg2, on="key", how="right")
merge_right_join

"""
Left anti-join
expected output: all the values that appear in the first data frame (left table),
but none of the matching values from the second data frame (right table).
"""

merge_left_anti = pd.merge(merg1, merg2, on="key", how="left", indicator=True)
merge_left_anti

merge_left_anti_join = merge_left_anti[merge_left_anti["_merge"] == "left_only"]
merge_left_anti_join

merge_left_anti_join = merge_left_anti_join.drop("_merge", axis=1)
# * in python axis=0 means rows and axis=1 means columns
merge_left_anti_join

"""
Right anti-join
expected output: all the values that appear in the second data frame (right table),
but none of the matching values from the first data frame (left table).
"""

merge_right_anti = pd.merge(merg1, merg2, on="key", how="right", indicator=True)
merge_right_anti

merge_right_anti_join = merge_right_anti[merge_right_anti["_merge"] == "right_only"]
merge_right_anti_join

merge_right_anti_join = merge_right_anti_join.drop("_merge", axis=1)
merge_right_anti_join


# * Part 1: Data Visualization in Python

x_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y_values = [1, 4, 6, 8, 10, 12, 14, 16, 18, 20]

plt.plot(x_values, y_values, color="green")
plt.xlabel("x axis placeholder")
plt.ylabel("y axis placeholder")
plt.title("Title")
plt.show()


plt.scatter(x_values, y_values, color="green")
plt.xlabel("x axis placeholder")
plt.ylabel("y axis placeholder")
plt.title("Title")
plt.show()


animal = ["cat", "dog", "horse", "mouse"]
animal_values = [10, 30, 100, 1]

plt.bar(animal, animal_values, color="forestgreen")
plt.xlabel("Animal")
plt.ylabel("Weight")
plt.title("Weight of Animals")
plt.show()


x_normal = np.random.normal(0, 1, 100)

plt.hist(x_normal, color="forestgreen")
plt.xlabel("x")
plt.ylabel("Probability")
plt.title("Randomly Sampled Data from Standard Normal Distribution")
plt.show()


# * Population distribution
# * np.arange(start, stop, step)
# * stats.norm generages the corresponding probability distribution values
x_population_values = np.arange(-4, 4, 0.01)
y_population_values = stats.norm.pdf(x_population_values)

count, bins, ignored = plt.hist(
    x_population_values, 30, density=True, color="purple", label="Sampling Distribution"
)

plt.plot(
    x_population_values,
    y_population_values,
    color="y",
    linewidth=2.5,
    label="Population Distribution",
)
plt.title("Randomly generating 100 obs from Normal distribution mu = 0 sigma = 1")
plt.ylabel("Probability")
plt.legend()
plt.show()
