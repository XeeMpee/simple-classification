import numpy as np
from enum import Enum
from typing import List

class RestorationType(Enum):
    MEAN = 0,


class DataRestorer:
    def __init__(self, restoration_type: RestorationType, malformed_tags: List) -> None:
        self.restoration_type = restoration_type
        self.malformed_tags = malformed_tags

    def restore(self, df, class_tag):
        if(self.restoration_type == RestorationType.MEAN):
            return self.__mean_restoration(df, class_tag)
        else:
            raise ValueError
        
    def __mean_restoration(self, df, class_tag):
        # purifying rows with malformed classification 
        for i,v in enumerate(df[class_tag]):
            if v in self.malformed_tags:
                df = df.drop(i,axis=0)
                
        tmpdf = df.drop(class_tag, axis=1)
        data = tmpdf.to_numpy()
        for malformed_tag in self.malformed_tags:
            data = np.select([data == malformed_tag],[np.nan], data)
        data = data.astype(np.float32)
            
        for row_indx, row in enumerate(data):
            clear_row = row[~np.isnan(row)]
            mean = np.mean(clear_row)
            row[np.isnan(row)] = mean
            data[row_indx] = row
        
        tmpdf.loc[:] = data
        df = tmpdf.assign(**{class_tag : df[class_tag]})
        return df
        