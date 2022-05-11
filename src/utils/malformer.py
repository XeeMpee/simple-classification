from random import random
from gevent import config
import pandas as pd


class Malformer:
    def __init__(self, probability: float) -> None:
        self.probability = probability

    def __decision(self):
        return random() < self.probability

    def malform(self, df: pd.DataFrame, class_tag: str):
        print("Malforming..")
        for i, column in enumerate(df.drop(class_tag, axis=1)):
            for j, item in enumerate(df[column]):
                if(self.__decision()):
                    df.loc[j:j, column] = None
        print("Malfroming done!")
        return df
