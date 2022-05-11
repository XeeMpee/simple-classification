import os
import glob
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

OTHERS_DIR = "docs/data_spec/others"
BOXPLOTS_DIR = "docs/data_spec/boxplots"
HISTOGRAMS_DIR = "docs/data_spec/histograms"

def data_description(df):
    print(df.describe())


def histograms(indx, label, data):
    plt.figure()
    plt.title(label)

    # histograms
    histogram = np.histogram(data, bins=100)
    plt.bar(histogram[1][:-1], histogram[0], width=1)
    plt.savefig(f'{HISTOGRAMS_DIR}/{indx}.png')
    plt.close()
    print(f"{label} histograms generated...")


def boxplots(indx, label, data):
    plt.figure()
    plt.title(label)

    plt.boxplot(data)
    plt.savefig(f'{BOXPLOTS_DIR}/{indx}.png')
    plt.close()
    print(f"{label} boxplots generated...")


def correlation_matrix(df):
    plt.figure(figsize=(20, 20))
    corr = df.corr()
    sb.heatmap(corr, cmap="Blues", annot=True)
    plt.savefig(f'{OTHERS_DIR}/correlation_matrix.png')
    print("correlation matrix generated...")


def importance_plot(class_tag, df, dfy):
    X = df.to_numpy()

    forest = RandomForestClassifier(random_state=0)
    forest.fit(X, dfy.values.ravel().astype(np.int8))
    importances = forest.feature_importances_

    forest_importances = pd.Series(
        importances, index=df.columns)

    fig, ax = plt.subplots(figsize=(8, 8))
    std = np.std(
        [tree.feature_importances_ for tree in forest.estimators_], axis=0)
    forest_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    plt.savefig(f'{OTHERS_DIR}/feature_importance.png')


# ================

def run(df, dfy, class_tag):

    print("Data spec generating...")
    for directory in [BOXPLOTS_DIR, HISTOGRAMS_DIR, OTHERS_DIR]:
        if os.path.exists(directory):
            shutil.rmtree(directory)
        os.mkdir(directory)

    for indx, label in enumerate(df):
        data = df[label].to_numpy(dtype=float)
        histograms(indx, label, data)
        boxplots(indx, label, data)

    correlation_matrix(df)
    importance_plot(class_tag, df, dfy)
    print("Generating data done")
