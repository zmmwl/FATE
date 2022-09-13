from fate.core.context import Context
from fate.interface.module import Module

from ..parser.data import Datasets
from .procedure import Procedure


class SkipRun(Procedure):
    @property
    def is_activate(self):
        return not self.situations.need_run

    def run(self, ctx: Context, cpn: Module, params, datasets: Datasets):
        if isinstance(datasets.data, dict) and len(datasets.data) >= 1:
            return list(datasets.data.values())[0]
