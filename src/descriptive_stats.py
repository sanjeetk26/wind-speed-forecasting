"""
To plot and compute various descriptive statistics pertaining to
wind speed and its correlation with other factors
"""

import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

non_factors = ["Year", "Month", "Day", "Hour", "Minute", "Unnamed: 18", "Snow Depth"]
regions = ["Rajasthan1", "Rajasthan2", "Rajasthan3", "Rajasthan4", "Rajasthan5"]
years = range(2000, 2015)


def get_df(region: str):
    data_path = f"../data/{region}/"

    # get all the data for a particular region
    all_files = glob.iglob(os.path.join(data_path, "*.csv"))
    df = (pd.read_csv(f, skiprows=2) for f in all_files)
    df = pd.concat(df, ignore_index=True)

    # sort by time
    sort_cols = ["Year", "Month", "Day", "Hour", "Minute"]
    df.sort_values(by=sort_cols, inplace=True)
    return df


def plot_hist(region: str, year: int):
    # make the necessary dirs to store the plot
    os.makedirs(f"../plots/histograms/{region}", exist_ok=True)

    df = get_df(region)
    df = df.loc[df["Year"] == year]

    # plot it
    sns.set_context("talk")
    plt.figure(figsize=(15, 12))
    plt.xlabel("Wind Speed")
    plt.title(f"Histogram for {year}")
    sns.histplot(df["Wind Speed"], kde=True)
    plt.savefig(f"../plots/histograms/{region}/{year}.png")


def plot_boxplot(region: str):
    os.makedirs(f"../plots/boxplot", exist_ok=True)

    df = get_df(region)
    sns.set_context("talk")
    plt.figure(figsize=(15, 12))
    plt.title(f"Box Plot - {region}")
    sns.boxplot(x="Wind Speed", y="Year", data=df, orient="h", whis=2, palette="muted")
    plt.savefig(f"../plots/boxplot/{region}.png")


# Plot all histograms
# for region in regions:
#     for yr in years:
#         plot_hist(region, yr)

# Plot all boxpplots
# for region in regions:
#     plot_boxplot(region)
