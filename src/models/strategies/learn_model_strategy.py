import abc
import numpy as np


class LearnModelStrategy(abc.ABC):
    def __init__() -> None:
        pass

    @abc.abstractclassmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    @abc.abstractclassmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
    
    @abc.abstractclassmethod
    def raw(self):
        raise NotImplementedError()