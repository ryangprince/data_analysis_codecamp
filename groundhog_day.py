# PRACTICE PROJECT

# * Groundhog Day - Data Science Project Live!
# * Link to YouTube: https://www.youtube.com/live/3niTFmVIjqQ?si=5p1WS4peORuY_5Uw
# * Link to Kaggle notebook: https://www.kaggle.com/datasets/groundhogclub/groundhog-day
# * Link to GROUNDHOG-DAY.COM APIs: https://groundhog-day.com/api


import pandas as pd
import numpy as np
import requests
import folium

df = pd.read_csv("data/raw/groundhog_archive.csv")
df.head()
df.tail()

df["Punxsutawney Phil"].value_counts()
df.loc[df["Punxsutawney Phil"] == "Partial Shadow"]


"""
Groundhog Dataset Creation

https://groundhog-day.com/api
/api/v1/groundhogs
"""

resp = requests.get("https://groundhog-day.com/api/v1/groundhogs")
hogs = pd.DataFrame(resp.json()["groundhogs"])
hogs = hogs.drop(["predictions"], axis=1)
hogs.to_csv("data/raw/hogs.csv", index=False)
hogs


# * Generate a map of groundhog locations

hogs["latitude"] = pd.to_numeric(hogs["coordinates"].str.split(",").str[0])
hogs["longitude"] = pd.to_numeric(hogs["coordinates"].str.split(",").str[1])

hogs["longitude"].values[0]

# ! mean() used to generate the map object, custom markets set specific locations

map = folium.Map(
    location=[hogs["latitude"].mean(), hogs["longitude"].mean()], zoom_start=3
)

for _, row in hogs.query("isGroundhog == 1").iterrows():
    folium.Marker(
        location=[row["latitude"], row["longitude"]],
        popup=row["slug"],
        icon=folium.Icon(color="green"),
    ).add_to(map)

for _, row in hogs.query("isGroundhog != 1").iterrows():
    folium.Marker(
        location=[row["latitude"], row["longitude"]],
        popup=row["slug"],
        icon=folium.Icon(color="blue"),
    ).add_to(map)

map

hogs["type"].value_counts().sort_values(ascending=True).plot(
    kind="barh", title="Count by Prognosticator Type"
)


# * Explode out Predictions from each Prognosticator

progs = pd.DataFrame(resp.json()["groundhogs"])
for i, d in progs.iterrows():
    break

df = pd.DataFrame(d["predictions"])

df.assign(**d[["id", "slug"]].to_dict())
