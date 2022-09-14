
from typing import List, Protocol


class Metric(Protocol):
    ...

class Metrics(Protocol):
    def log(self, name: str, namespace: str, metric_data: List[Metric]):
        ...
