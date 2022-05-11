import pandas as pd
import sklearn.svm

from src.models.learn_model import LearnModel

class SvmModel(LearnModel):
    def __init__(self, class_tag: str, df: pd.DataFrame) -> None:
        LearnModel.__init__(self, class_tag, df)
        self.model = sklearn.svm.SVC()

    def train(self):
        pass

    def predict(self, sample):
        pass
