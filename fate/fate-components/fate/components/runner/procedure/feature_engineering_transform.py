from fate.core.context import Context
from fate.interface.module import Module

from ..parser.data import Datasets
from .procedure import Procedure


class FeatureEngineeringTransform(Procedure):
    def is_activate(self):
        return self.situations.has_data and self.situations.has_model

    def run(self, ctx: Context, cpn: Module, params, datasets: Datasets, models):
        if self.has_model:
            cpn.load_model(models)
        data = cpn.extract_data(datasets.data)
        with ctx.namespace("transform") as subctx:
            return cpn.transform(subctx, data)
