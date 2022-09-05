from federatedml.model_base import ModelBase
from federatedml.util import LOGGER
from federatedml.custom_nn.fedavg_scheduler import FedAvgSchedulerAggregator
from .param import NNParam


class NNAggregator(ModelBase):
    def __init__(self):
        super().__init__()
        self.model_param = NNParam()

    def _init_model(self, model):
        return super()._init_model(model)

    def run(self, cpn_input, retry: bool = True):
        epochs = 2
        fedavg = FedAvgSchedulerAggregator()

        for epoch in range(1, epochs + 1):
            LOGGER.info(f"epoch {epoch} start")
            fedavg.step()
