from typing import List
from src.models.models.i_learn_model import ILearnModel
class LearnModelFactory:
    
    def __init__(self) -> None:
        self.models = {
            "voting-classifier-soft": SvmModel(),
            "voting-classifier-hard":  RandomForestModel(),
            "stacking-classifier":  RandomForestModel()
        }
    
    def create(self, model_name : str, estimators: List[ILearnModel]) -> ILearnModel:
        return self.models.get(model_name)()
        