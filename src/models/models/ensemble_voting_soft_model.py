import numpy as np
from typing import List
from sklearn.ensemble import VotingClassifier

from src.models.models.i_learn_model import ILearnModel


class EnsembleVotingSoftClassifierModel(ILearnModel):
    def __init__(self, models: List[ILearnModel]) -> None:
        self.clf = VotingClassifier(
                        estimators=[model.raw() for model in models],
                        voting='soft'
                    )

    def fit(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.clf.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict(X)

    def raw(self):
        return self.clf