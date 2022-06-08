from sklearn import metrics
from src.config.config import Config


class ModelMetrics:
    def __init__(self, config: Config, y, y_predicted) -> None:
        self.config = config
        self.y = y
        self.y_predicted = y_predicted

    def all_metrics(self):
        metrics = []
        for metrics_name in self.config.metrics:
            metrics.append((metrics_name, self._metrics(
                metrics_name, self.y, self.y_predicted)))
        return metrics

    def major_metrics(self):
        return self._metrics(self.config.metrics[0], self.y, self.y_predicted)

    def _metrics(self, metrics_name, y, y_predicted):
        if(metrics_name == "accuracy"):
            return metrics.accuracy_score(y, y_predicted)
        if(metrics_name == "confusion-matrix"):
            return metrics.confusion_matrix(y, y_predicted)
        if(metrics_name == "recall"):
            return metrics.recall_score(y, y_predicted)
        if(metrics_name == "precision"):
            return metrics.precision_score(y, y_predicted)
        if(metrics_name == "f1"):
            return metrics.f1_score(y, y_predicted)

        raise IndexError("No such metrics")
