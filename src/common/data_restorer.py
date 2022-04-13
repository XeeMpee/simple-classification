import numpy as np
from enum import Enum


class RestorationType(Enum):
    MEAN = 0,


class DataRestorer:
    def __init__(self, restoration_type: RestorationType) -> None:
        self.restoration_type = restoration_type

    def restore(self, df):
        if(self.restoration_type == RestorationType.MEAN):
            self.__mean_restoration(df)
        else:
            raise ValueError
        
    def __mean_restoration(self, df):
        for index, column in enumerate(df):
            data = df[column].to_numpy().astype(np.float64)
            clear_data = data[~np.isnan(data)]
            mean = np.mean(clear_data)
            data[np.isnan(data)] = mean
            df.loc[:,column] = data 