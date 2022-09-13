from fate.core.context import Context
from fate.interface.module import Module

from ..parser.data import Datasets
from .procedure import Procedure
from ..utils import set_predict_data_schema, union_data


class TrainWithoutValidate(Procedure):
    def is_activate(self):
        return self.situations.has_train_data and (
            not self.situations.has_validate_data
        )

    def run(self, ctx: Context, cpn: Module, params, datasets: Datasets, models):
        if self.has_model:
            cpn.load_model(models)
        with ctx.namespace("fit") as subctx:
            cpn.fit(subctx, datasets.train_data)
        with ctx.namespace("predict") as subctx:
            predict_on_train_data = cpn.predict(subctx, datasets.validate_data)
        union_output = union_data([predict_on_train_data], ["train"])
        return set_predict_data_schema(union_output, datasets.schema)
