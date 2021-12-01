"""
To plot and compute various descriptive statistics pertaining to
wind speed and its correlation with other factors
"""

import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import numpy as np
from factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
from factor_analyzer import FactorAnalyzer

regions = ["Rajasthan1", "Rajasthan2", "Rajasthan3", "Rajasthan4", "Rajasthan5"]
years = range(2000, 2015)
non_factors = ["Year", "Month", "Day", "Hour", "Minute", "Unnamed: 18", "Snow Depth"]


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


def plot_corr_map(region: str):
    os.makedirs(f"../plots/factor-analysis/corr", exist_ok=True)

    df = get_df(region)
    df.drop(columns=non_factors, inplace=True)
    corr_map = df.corr()

    # mask to hide upper triangle
    mask = np.zeros_like(corr_map)
    tri_indices = np.triu_indices_from(mask, k=1)
    mask[tri_indices] = True

    sns.set_context("talk")
    plt.figure(figsize=(15, 12))
    sns.heatmap(
        corr_map,
        annot=True,
        mask=mask,
        cmap=sns.diverging_palette(300, 145, s=60, as_cmap=True),
        annot_kws={"size": 14},
    )
    sns.set_style("white")
    plt.savefig(f"../plots/factor-analysis/corr/{region}.png", bbox_inches="tight")


def calc_bts(region: str):
    df = get_df(region)
    df.drop(columns=non_factors, inplace=True)
    print(calculate_bartlett_sphericity(df))


def calc_kmo(region: str):
    df = get_df(region)
    df.drop(columns=non_factors, inplace=True)
    print(calculate_kmo(df)[1])


def scree_plot(region: str):
    os.makedirs(f"../plots/factor-analysis/scree", exist_ok=True)

    df = get_df(region)
    df.drop(columns=non_factors, inplace=True)

    # get the eigenvalues
    fa = FactorAnalyzer(n_factors=12, rotation=None)
    fa.fit(df)
    evalues, evectors = fa.get_eigenvalues()

    # plot
    sns.set_context("talk")
    plt.figure(figsize=(15, 12))
    xvals = range(1, 13)
    sns.scatterplot(x=xvals, y=evalues)
    sns.lineplot(x=range(1, 13), y=evalues)
    plt.xlabel("Factors")
    plt.ylabel("Eigenvalues")
    plt.title("Scree Plot")
    plt.xticks(xvals)
    plt.savefig(f"../plots/factor-analysis/scree/{region}.png")


def fa_loadings(region: str):
    df = get_df(region)
    df.drop(columns=non_factors, inplace=True)

    # perform factor analysis
    fa = FactorAnalyzer(n_factors=3, rotation="varimax")
    fa.fit(df)

    loadings_df = pd.DataFrame.from_records(
        fa.loadings_, columns=["Factor 1", "Factor 2", "Factor 3"], index=df.columns
    )
    print(loadings_df)


def fa_variance(region: str):
    df = get_df(region)
    df.drop(columns=non_factors, inplace=True)

    # perform factor analysis
    fa = FactorAnalyzer(n_factors=3, rotation="varimax")
    fa.fit(df)

    loadings_df = pd.DataFrame.from_records(
        fa.get_factor_variance(),
        columns=["Factor 1", "Factor 2", "Factor 3"],
        index=["SS Loadings", "Proportion Variance", "Cumulative Variance"],
    )
    print(loadings_df)


# Plot all histograms
# for region in regions:
#     for yr in years:
#         plot_hist(region, yr)

# Plot all boxpplots
# for region in regions:
#     plot_boxplot(region)

# for region in regions:
#     plot_corr_map(region)

# for r in regions:
#     scree_plot(r)

# fa_variance(regions[0])
