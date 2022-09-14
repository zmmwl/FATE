from typing import List
from fate.interface import Metrics as MetricsInterface
from fate.interface import Metric as MetricInterface

class Metric(MetricInterface):
    ...

class Metrics(MetricsInterface):
    def __init__(self, ctx) -> None:
        self.ctx = ctx
        self.name_surfix = None

    def log(self, name: str, namespace: str, metric_data: List[Metric]):
        if self.name_surfix is not None:
            name = f"{name}{self.name_surfix}"
        self.ctx.tracker.log_metric_data(
            metric_name=name,
            metric_namespace=namespace,
            metrics=metric_data,
        )
