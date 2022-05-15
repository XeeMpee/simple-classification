import sklearn.svm
import numpy as np

from src.models.strategies.learn_model_strategy import LearnModelStrategy

class SvmModelStrategy(LearnModelStrategy):
    def __init__(self) -> None:
        self.clf = sklearn.svm.SVC(kernel="rbf")

    def fit(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.clf.fit(X,y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict(X)
