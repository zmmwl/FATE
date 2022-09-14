from contextlib import contextmanager
from typing import Iterator, Protocol

from ._summary import Summary
from ._cache import Cache
from ._metric import Metrics
from ._anonymous import Anonymous
from ._checkpoint import CheckpointManager


class Context(Protocol):
    role: str
    party_id: str
    summary: Summary
    metrics: Metrics
    cache: Cache
    anonymous_generator: Anonymous
    checkpoint_manager: CheckpointManager

    @contextmanager
    def sub_ctx(self, namespace) -> Iterator["Context"]:
        ...
