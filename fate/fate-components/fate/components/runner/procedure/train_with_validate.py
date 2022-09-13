from fate.core.context import Context
from fate.interface.module import Module

from ..parser.data import Datasets
from .procedure import Procedure
from ..utils import set_predict_data_schema, union_data


class TrainWithValidate(Procedure):
    def is_activate(self):
        return self.situations.has_train_data and self.situations.has_validate_data

    def run(self, ctx: Context, cpn: Module, params, datasets: Datasets):
        if self.has_model:
            cpn.load_model(models)
        with ctx.namespace("fit") as subctx:
            cpn.fit(subctx, datasets.train_data, datasets.validate_data)
        with ctx.namespace("predict") as subctx:
            predict_on_validate_data = cpn.predict(subctx, datasets.train_data)
        with ctx.namespace("validate") as subctx:
            predict_on_train_data = cpn.predict(subctx, datasets.validate_data)
        union_output = union_data(
            [predict_on_validate_data, predict_on_train_data], ["train", "validate"]
        )
        return set_predict_data_schema(union_output, datasets.schema)
