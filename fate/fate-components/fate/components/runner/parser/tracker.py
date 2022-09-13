import typing


class MetricType:
    LOSS = "LOSS"


class Metric:
    def __init__(self, key, value: float, timestamp: float = None):
        self.key = key
        self.value = value
        self.timestamp = timestamp

    def to_dict(self):
        return dict(key=self.key, value=self.value, timestamp=self.timestamp)


class MetricMeta:
    def __init__(self, name: str, metric_type: MetricType, extra_metas: dict = None):
        self.name = name
        self.metric_type = metric_type
        self.metas = {}
        self.extra_metas = extra_metas

    def update_metas(self, metas: dict):
        self.metas.update(metas)

    def to_dict(self):
        return dict(
            name=self.name,
            metric_type=self.metric_type,
            metas=self.metas,
            extra_metas=self.extra_metas,
        )

class Tracker:
    def __init__(self, tracker) -> None:
        self._tracker = tracker

    @classmethod
    def parse(cls, tracker):
        return Tracker(tracker)

    def log_metric_data(
        self, metric_namespace: str, metric_name: str, metrics: typing.List[Metric]
    ):
        return self._tracker.log_metric_data(
            metric_namespace=metric_namespace,
            metric_name=metric_name,
            metrics=[metric.to_dict() for metric in metrics],
        )

    def set_metric_meta(
        self, metric_namespace: str, metric_name: str, metric_meta: MetricMeta
    ):
        return self._tracker.set_metric_meta(
            metric_namespace=metric_namespace,
            metric_name=metric_name,
            metric_meta=metric_meta.to_dict(),
        )

    def log_component_summary(self, summary_data: dict):
        return self._tracker.log_component_summary(summary_data=summary_data)
