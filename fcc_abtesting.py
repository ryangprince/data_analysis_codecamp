# PRACTICE PROJECT

# * Python for Data Science Course – Hands-on Projects with EDA, AB Testing & Business Intelligence
# * Link to YouTube: https://youtu.be/FTpmwX94_Yo?si=vw-3m3265jBP2ESB
# * Link to 'percent_bachelors_women_usa.csv': https://github.com/johnashu/datacamp/blob/master/percent-bachelors-degrees-women-usa.csv
# * Link to GitHub repo for Project 1: https://github.com/TatevKaren/CaseStudies/tree/main/AB%20Testing

# * PROJECT 1


import numpy as np
import pandas as pd
from scipy.stats import norm
import seaborn as sns
import matplotlib.pyplot as plt


df_ab_test = pd.read_csv("data/raw/ab_test_click_data.csv")
df_ab_test.head()
df_ab_test.describe()
df_ab_test.groupby("group").sum("click")


# * Custom palette for yellow and black
palette = {0: "yellow", 1: "black"}  # * Assuming 0 for no click and 1 for click

# * Plotting the click distribution for each group with the custom colors
plt.figure(figsize=(10, 6))
ax = sns.countplot(x="group", hue="click", data=df_ab_test, palette=palette)
plt.title("Click distribution in Experimental and Control Groups")
plt.xlabel("Group")
plt.ylabel("Count")
plt.legend(title="Click", labels=["No", "Yes"])

# * Calculate percentages and annotate bars
group_counts = df_ab_test.groupby(["group"]).size()
group_count_clicks = (
    df_ab_test.groupby(["group", "click"]).size().reset_index(name=("count"))
)

for p in ax.patches:
    height = p.get_height()
    # * Find the group and click type for the current bar
    group = "exp" if p.get_x() < 0.5 else "con"
    click = 1 if p.get_x() > 0.5 else 0
    total = group_counts.loc[group]
    percentage = 100 * height / total
    ax.text(
        p.get_x() + p.get_width() / 2.0,
        height + 5,
        f"{percentage:.1f}%",
        ha="center",
        color="black",
        fontsize=10,
    )

plt.tight_layout()
plt.show()


# * Parameters of Model from Power Analysis

alpha = 0.05
print(f"Alpha: significance level is: {alpha}")

delta = 0.1
print(f"\nDelta: minimum detectable effect is: {delta}")
