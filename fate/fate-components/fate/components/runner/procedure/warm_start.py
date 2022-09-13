from fate.core.context import Context
from fate.interface.module import Module

from ..parser.data import Datasets
from .procedure import Procedure


class WarmStart(Procedure):

    def run(self, ctx: Context, cpn: Module, params, datasets: Datasets, models):
        if self.has_model:
            cpn.load_model(models)
        ...
