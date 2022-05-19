import sklearn.svm
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.models.strategies.learn_model_strategy import LearnModelStrategy

class RandomForestModelStrategy(LearnModelStrategy):
    def __init__(self) -> None:
        self.clf = RandomForestClassifier(max_depth=4, random_state=0)

    def fit(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.clf.fit(X,y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict(X)

    def raw(self):
        return self.clf