from fate.core.context import Context as ContextBase
from fate.core.context import Namespace

from .parser.anonymous import Anonymous
from .parser.cache import Cache
from .parser.checkpoint import CheckpointManager
from .parser.metric import Metrics
from .parser.summary import Summary


class ComponentContext(ContextBase):
    """
    implement fate.interface.Context for flow runner

    this implemention is specificated for fate.flow, ie:
      - `summary` and `metrics` are traceback using flow's track client
      - `metrics` has additional propety `surfix` for CV/Stepwise to log metric separated in fold
      - ...
    """

    def __init__(
        self,
        role: str,
        party_id: str,
        tracker,
        checkpoint_manager: CheckpointManager,
        namespace: Namespace,
    ) -> None:
        self.namespace = namespace
        self.role = role
        self.party_id = party_id

        self.tracker = tracker
        self.summary: Summary = Summary(self)
        self.metrics: Metrics = Metrics(self)
        self.cache: Cache = Cache()
        self.anonymous_generator: Anonymous = Anonymous(role, party_id)
        self.checkpoint_manager: CheckpointManager = checkpoint_manager
