from fate.core.context import Context
from fate.interface.module import Module
from federatedml.model_selection.k_fold import KFold

from ..parser.data import Datasets
from .procedure import Procedure


class CrossValidation(Procedure):

    @property
    def is_activate(self):
        return self.situations.has_model and self.situations.has_train_data

    def run(self, ctx: Context, cpn: Module, params, datasets: Datasets):
        kflod_obj = KFold()
        params.cv_param.role = ctx.role
        params.cv_param.mode = cpn.mode
        output_data = kflod_obj.run(params.cv_param, datasets.train_data, cpn, False)
        return output_data
