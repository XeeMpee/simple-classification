import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.common.class_tags_converter import ClassLabelsConverter

redundant_columns = ["obj_ID", "class"]
rows_malformed_values = ["#NUM!"]
class_tag = ["is_safe"]

def column_purification(df):
    for label in redundant_columns:
        if(label in df.columns):
            df.pop(label)
    print("columns purified...")


def rows_purification(df: pd.DataFrame):
    malformed_rows = []
    for column in df.columns:
        for malform_value in rows_malformed_values:
            malformed_rows += df.index[df[column] == malform_value].tolist()
    for row in malformed_rows:
        try:
            df.drop(row, axis=0, inplace=True)
        except:
            pass
    print("rows purified...")


def data_description(df):
    print(df.describe())


def histograms(indx, label, data):
    plt.figure()
    plt.title(label)

    # histograms
    histogram = np.histogram(data, bins=100)
    plt.bar(histogram[1][:-1], histogram[0], width=1)
    plt.savefig(f'docs/data_spec/histograms/{indx}.png')
    plt.close()
    print(f"{label} histograms generated...")


def boxplots(indx, label, data):
    plt.figure()
    plt.title(label)

    plt.boxplot(data)
    plt.savefig(f'docs/data_spec/boxplots/{indx}.png')
    plt.close()
    print(f"{label} boxplots generated...")


def correlation_matrix(df):
    plt.figure(figsize=(20, 20))
    corr = df.corr()
    sb.heatmap(corr, cmap="Blues", annot=True)
    plt.savefig(f'docs/data_spec/others/correlation_matrix.png')
    print("correlation matrix generated...")

def importance_plot(df):
    X = df.drop(class_tag, axis=1).to_numpy()
    y= df[class_tag].values
    labelsConvereter = ClassLabelsConverter(y)
    y = labelsConvereter.convertToIdentifiers(y)
    
    forest = RandomForestClassifier(random_state=0)
    # X_train, X_test, y_train, y_test = train_test_split(X, y.ravel(), stratify=y, random_state=42)
    forest.fit(X, y.ravel())
    importances = forest.feature_importances_
    
    forest_importances = pd.Series(importances, index=df.columns.drop(class_tag))

    fig, ax = plt.subplots(figsize=(8,8))
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    forest_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    plt.savefig('docs/data_spec/others/feature_importance.png')
    
    

# ================

def run():
    df = pd.read_csv("data/waterQuality1.csv")
    column_purification(df)
    rows_purification(df)

    for indx, label in enumerate(df):
        data = df[label].to_numpy(dtype=float)
        histograms(indx, label, data)
        boxplots(indx, label, data)
        
    correlation_matrix(df)
    importance_plot(df)
    

    print("DONE")
