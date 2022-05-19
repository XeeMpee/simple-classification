from src.models.learn_model import LearnModel
from src.models.strategies.svm_model_strategy import  SvmModelStrategy
from src.models.strategies.random_forest_model_strategy import RandomForestModelStrategy
class LearnModelFactory:
    
    def __init__(self) -> None:
        self.models = {
            "svm": lambda : self._model(SvmModelStrategy()),
            "random-forest": lambda : self._model(RandomForestModelStrategy())
        }
    
    def create(self, model_name : str) -> LearnModel:
        return self.models.get(model_name)()
        
        
    def _model(self, strategy) -> LearnModel:
        model = LearnModel()
        model.set_strategy(strategy)
        return model
        