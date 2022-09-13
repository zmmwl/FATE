from fate.core.context import Context
from fate.interface.module import Module

from ..parser.data import Datasets
from .procedure import Procedure
from ..utils import set_predict_data_schema, union_data


class Predict(Procedure):
    def is_activate(self):
        return (not self.situations.has_test_data) and self.situations.has_test_data

    def run(self, ctx: Context, cpn: Module, params, datasets: Datasets, models):
        if self.has_model:
            cpn.load_model(models)
        with ctx.namespace("predict") as subctx:
            predict_on_test_data = cpn.predict(subctx, datasets.test_data)
        union_output = union_data([predict_on_test_data], ["predict"])
        return set_predict_data_schema(union_output, datasets.schema)
