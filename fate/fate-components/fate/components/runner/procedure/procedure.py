from fate.components.runner.runner import dataclasses
from fate.core.context import Context
from fate.interface.module import Module

from ..parser.data import Datasets


@dataclasses.dataclass
class Situation:
    need_run: bool
    need_cv: bool
    need_stepwise: bool
    has_model: bool
    has_isometric_model: bool
    has_train_data: bool
    has_test_data: bool
    has_validate_data: bool
    has_data: bool


class Procedure:
    def __init__(self, situations: Situation) -> None:
        self.situations = situations

    @property
    def is_activate(self):
        return False

    @property
    def has_model(self):
        return self.situations.has_model or self.situations.has_isometric_model

    def run(cls, ctx: Context, cpn: Module, params, datasets: Datasets, models):
        ...
