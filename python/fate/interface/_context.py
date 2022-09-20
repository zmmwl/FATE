from contextlib import contextmanager
from typing import Iterator, Protocol

from ._anonymous import Anonymous
from ._cache import Cache
from ._checkpoint import CheckpointManager
from ._log import Logger
from ._metric import Metrics
from ._summary import Summary
from ._party import Party, Parties

class Context(Protocol):
    role: str
    party_id: str
    summary: Summary
    metrics: Metrics
    cache: Cache
    anonymous_generator: Anonymous
    checkpoint_manager: CheckpointManager
    log: Logger
    guest: Party
    hosts: Parties
    arbiter: Party

    @contextmanager
    def sub_ctx(self, namespace) -> Iterator["Context"]:
        ...
