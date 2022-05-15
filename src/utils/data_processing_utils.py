import pandas as pd
import numpy as np
from sklearn import model_selection
from random import random
from dataclasses import dataclass

@dataclass
class Dataset:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    

class DataProcessingUtils:
    def __init__(self,) -> None:
        pass

    def __decision(self, probability):
        return random() < probability

    def malform(self, df: pd.DataFrame, class_tag: str,  probability: float):
        print("Malforming..")
        for i, column in enumerate(df.drop(class_tag, axis=1)):
            for j, item in enumerate(df[column]):
                if(self.__decision(probability)):
                    df.loc[j:j, column] = None
        print("Malfroming done!")
        return df

    def divide(self, df: pd.DataFrame, class_tag: str, ratio: float) -> Dataset:
        y = df[class_tag].to_numpy().astype(np.int64)
        X = df.drop(class_tag, axis=1).to_numpy().astype(np.int64)
        return Dataset(*model_selection.train_test_split(X, y, test_size=ratio))
