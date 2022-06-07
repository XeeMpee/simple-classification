import numpy as np
from typing import List
from sklearn.ensemble import VotingClassifier

from src.models.models.i_learn_model import ILearnModel


class EnsembleVotingHardClassifierModel(ILearnModel):
    def __init__(self, estimators) -> None:
        self.clf = VotingClassifier(
            estimators=estimators,
            voting='hard'
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.clf.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict(X)

    def raw(self):
        return self.clf

    def name(self):
        return "EnsembleVotingHardClassifierModel"
