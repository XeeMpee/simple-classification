import sklearn.svm
import numpy as np

from src.models.models.i_learn_model import ILearnModel


class SvmModel(ILearnModel):
    def __init__(self) -> None:
        self.clf = sklearn.svm.SVC(kernel="sigmoid", probability=True)

    def fit(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.clf.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict(X)

    def raw(self):
        return self.clf

    def name(self):
        return "SvmModel"