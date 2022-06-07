from src.models.models.i_learn_model import ILearnModel
from src.models.models.svm_model import SvmModel
from src.models.models.random_forest_model import RandomForestModel


class LearnModelFactory:

    def __init__(self) -> None:
        self.models = {
            "svm": lambda: SvmModel(),
            "random-forest": lambda: RandomForestModel()
        }

    def create(self, model_name: str) -> ILearnModel:
        return self.models.get(model_name)()
