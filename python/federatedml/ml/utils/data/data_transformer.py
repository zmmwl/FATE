from federatedml.ml.data.data_transformer_param import DataTransformerParam
from federatedml.model_base import ComponentOutput, ModelBase
from fate_arch.tensor import Context
from fate_arch.utils.datasets import load_data_from_table


class DataTransformer(ModelBase):
    def __init__(self):
        super(DataTransformer, self).__init__()
        self.model_param = DataTransformerParam()

    def _init_runtime_parameters(self, cpn_input):
        self.model_param.update(cpn_input.parameters)

    def run(self, cpn_input):
        ctx = Context.from_cpn_input(cpn_input)
        self._init_runtime_parameters(cpn_input)

        data = cpn_input["data"]["data"]

        return load_data_from_table.load(ctx, data, **cpn_input.parameters)


