import sklearn.svm
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.models.models.i_learn_model import ILearnModel

class RandomForestModel(ILearnModel):
    def __init__(self) -> None:
        self.clf = RandomForestClassifier(max_depth=4, random_state=0)

    def fit(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.clf.fit(X,y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict(X)

    def raw(self):
        return self.clf