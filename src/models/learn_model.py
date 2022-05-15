import pandas as pd
import numpy as np
from src.config.config import Config
from src.models.strategies.learn_model_strategy import LearnModelStrategy
from src.utils.data_processing_utils import DataProcessingUtils
class LearnModel:
    
    def __init__(self) -> None:
        self.strategy = None
    
    def set_strategy(self, strategy: LearnModelStrategy):
        self.strategy = strategy
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        if not self.strategy:
            raise Exception("Learn model not set!")
        self.strategy.fit(X,y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.strategy:
            raise Exception("Learn model not set!")
        return self.strategy.predict(X)
