from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import List, Optional

from fate.interface import Anonymous, Cache, CheckpointManager
from fate.interface import Context as ContextInterface
from fate.interface import Metric as MetricInterface
from fate.interface import MetricMeta as MetricMetaInterface
from fate.interface import Metrics, Summary


@dataclass
class Metric(MetricInterface):
    key: str
    value: float
    timestamp: Optional[float] = None


class MetricMeta(MetricMetaInterface):
    def __init__(self, name: str, metric_type: str, extra_metas: Optional[dict] = None):
        self.name = name
        self.metric_type = metric_type
        self.metas = {}
        self.extra_metas = extra_metas

    def update_metas(self, metas: dict):
        self.metas.update(metas)


class DummySummary(Summary):
    """
    dummy summary save nowhre
    """

    def __init__(self) -> None:
        self._summary = {}

    @property
    def summary(self):
        return self._summary

    def save(self):
        pass

    def reset(self, summary: dict):
        self._summary = summary

    def add(self, key: str, value):
        self._summary[key] = value


class DummyMetrics(Metrics):
    def __init__(self) -> None:
        self._data = []
        self._meta = []

    def log(self, name: str, namespace: str, data: List[Metric]):
        self._data.append((name, namespace, data))

    def log_meta(self, name: str, namespace: str, meta: MetricMeta):
        self._meta.append((name, namespace, meta))

    def log_warmstart_init_iter(self, iter_num):  # FIXME: strange here
        ...


class DummyCache(Cache):
    def __init__(self) -> None:
        self.cache = []

    def add_cache(self, key, value):
        self.cache.append((key, value))


# FIXME: vary complex to use, may take times to fix
class DummyAnonymous(Anonymous):
    ...


class DummyCheckpointManager(CheckpointManager):
    ...


class Namespace:
    """
    Summary, Metrics may be namespace awared:
    ```
    namespace = Namespace()
    ctx = Context(...summary=XXXSummary(namespace))
    ```
    """

    def __init__(self) -> None:
        self.namespaces = []

    @contextmanager
    def into_subnamespace(self, subnamespace: str):
        self.namespaces.append(subnamespace)
        try:
            yield self
        finally:
            self.namespaces.pop()

    @property
    def namespace(self):
        return ".".join(self.namespaces)


class Context(ContextInterface):
    """
    implement fate.interface.ContextInterface

    Note: most parameters has default dummy value,
          which is convenient when used in script.
          please pass in custom implements as you wish
    """

    def __init__(
        self,
        role: str,
        party_id: str,
        summary: Summary = DummySummary(),
        metrics: Metrics = DummyMetrics(),
        cache: Cache = DummyCache(),
        anonymous_generator: Anonymous = DummyAnonymous(),
        checkpoint_manager: CheckpointManager = DummyCheckpointManager(),
        namespace: Optional[Namespace] = None,
    ) -> None:
        if namespace is None:
            self.namespace = Namespace()
        else:
            self.namespace = namespace

        self.role = role
        self.party_id = party_id
        self.summary = summary
        self.metrics = metrics
        self.cache = cache
        self.anonymous_generator = anonymous_generator
        self.checkpoint_manager = checkpoint_manager

    @contextmanager
    def sub_ctx(self, namespace) -> Iterator["Context"]:
        with self.namespace.into_subnamespace(namespace):
            yield self
