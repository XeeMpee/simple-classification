from random import random
import pandas as pd


class Malformer:
    def __init__(self, probability: float) -> None:
        self.probability = probability

    def __decision(self):
        return random() < self.probability

    def malform(self, df: pd.DataFrame):
        for i, column in enumerate(df.columns):
            for j, item in enumerate(df[column]):
                if(self.__decision()):
                    df.loc[j:j, column] = None