import pandas as pd
import sklearn.model_selection

class LearnModel:
    def __init__(self, class_tag: str, df: pd.DataFrame) -> None:
        self.df = df
        self.X = df.drop(class_tag, axis=1).to_numpy()
        self.y = df[class_tag].to_numpy()
        
        self.X_train, self.X_test, self.y_train, self.y_test = sklearn.model_selection.train_test_split(self.X, self.y, test_size=0.2)
    