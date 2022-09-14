from collections.abc import Iterator
from contextlib import contextmanager
from typing import List, Optional

from fate.interface import Anonymous, Cache, CheckpointManager
from fate.interface import Context as ContextInterface
from fate.interface import Metric, Metrics, Summary


class DummySummary(Summary):
    """
    dummy summary save nowhre
    """

    def save(self):
        pass


class DummyMetrics(Metrics):
    def log(self, name: str, namespace: str, metric_data: List[Metric]):
        pass


class DummyCache(Cache):
    def __init__(self) -> None:
        self.cache = []

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
