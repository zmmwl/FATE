import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from logging import Logger, getLogger
from typing import List, Literal, Optional, Tuple

from fate.interface import LOGMSG, Anonymous, Cache, CheckpointManager
from fate.interface import Context as ContextInterface
from fate.interface import Logger as LoggerInterface
from fate.interface import Metric as MetricInterface
from fate.interface import MetricMeta as MetricMetaInterface
from fate.interface import Metrics, Summary

from ._federation import GC, _PartyUtil
from ._namespace import Namespace


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


class DummyLogger(LoggerInterface):
    def __init__(self, level=logging.DEBUG) -> None:
        self.logger = getLogger("fate.dummy")
        self.logger.setLevel(level)

        # console
        formatter = logging.Formatter(
            "%(asctime)s - %(pathname)s:%(lineno)s - %(levelname)s - %(message)s"
        )
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)

        self.logger.addHandler(console_handler)

    def log(self, level: int, msg: LOGMSG):
        if Logger.isEnabledFor(self.logger, level):
            if callable(msg):
                msg = msg()
            self.logger.log(level, msg, stacklevel=3)

    def info(self, msg: LOGMSG):
        return self.log(logging.INFO, msg)

    def debug(self, msg: LOGMSG):
        return self.log(logging.INFO, msg)

    def error(self, msg: LOGMSG):
        return self.log(logging.INFO, msg)

    def warning(self, msg: LOGMSG):
        return self.log(logging.INFO, msg)


class Context(ContextInterface):
    """
    implement fate.interface.ContextInterface

    Note: most parameters has default dummy value,
          which is convenient when used in script.
          please pass in custom implements as you wish
    """

    def __init__(
        self,
        local_party: Tuple[Literal["guest", "host", "arbiter"], str],
        parties: Optional[List[Tuple[Literal["guest", "host", "arbiter"], str]]] = None,
        summary: Summary = DummySummary(),
        metrics: Metrics = DummyMetrics(),
        cache: Cache = DummyCache(),
        anonymous_generator: Anonymous = DummyAnonymous(),
        checkpoint_manager: CheckpointManager = DummyCheckpointManager(),
        log: DummyLogger = DummyLogger(),
        namespace: Optional[Namespace] = None,
    ) -> None:
        if namespace is None:
            self.namespace = Namespace()
        else:
            self.namespace = namespace

        self.role, self.party_id = local_party
        self.summary = summary
        self.metrics = metrics
        self.cache = cache
        self.anonymous_generator = anonymous_generator
        self.checkpoint_manager = checkpoint_manager
        self.log = log

        self._party_util = _PartyUtil.parse(local_party, parties)
        self._gc = GC()

    @contextmanager
    def sub_ctx(self, namespace) -> Iterator["Context"]:
        with self.namespace.into_subnamespace(namespace):
            yield self

    @property
    def guest(self):
        self._party_util.create_party("guest", self, self.namespace, self._gc)

    @property
    def hosts(self):
        self._party_util.create_parties("host", self, self.namespace, self._gc)

    @property
    def arbiter(self):
        self._party_util.create_party("arbiter", self, self.namespace, self._gc)

    @property
    def parties(self):
        return self._party_util.all_parties(self, self.namespace, self._gc)
