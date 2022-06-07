from typing import List, Tuple
from src.models.models.ensemble_stacking_model import EnsembleStackingClassifierModel
from src.models.models.ensemble_voting_hard_model import EnsembleVotingHardClassifierModel
from src.models.models.ensemble_voting_soft_model import EnsembleVotingSoftClassifierModel
from src.models.models.i_learn_model import ILearnModel


class EnsembleLearnModelFactory:

    def __init__(self) -> None:
        self.models = {
            "voting-classifier-soft": lambda estimators: EnsembleVotingSoftClassifierModel(estimators),
            "voting-classifier-hard": lambda estimators: EnsembleVotingHardClassifierModel(estimators),
            "stacking-classifier": lambda estimators: EnsembleStackingClassifierModel(estimators)
        }

    def create(self, model_name: str, models: List[Tuple[str, ILearnModel]]) -> ILearnModel:
        return self.models.get(model_name)([(models[i].name(), models[i].raw()) for i in range(0, len(models))])
