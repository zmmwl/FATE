import torch
from torch.optim import Optimizer
from federatedml.transfer_variable.base_transfer_variable import BaseTransferVariables


class FedAvgSchedulerTransferVariable(BaseTransferVariables):
    def __init__(self, flowid=0):
        super().__init__(flowid)
        self.client_params = self._create_variable(
            "client_params", src=["guest", "host"], dst=["arbiter"]
        )
        self.aggregated_params = self._create_variable(
            "aggregated_params", dst=["guest", "host"], src=["arbiter"]
        )


class FedAvgSchedulerClient:
    def __init__(self, optimizer: Optimizer):
        self.optimizer = optimizer
        self.transfer_variable = FedAvgSchedulerTransferVariable()
        self._step = 0

    def step(self):
        # get optimizer params
        params_group = [[p.detach().numpy() for p in group["params"]] for group in self.optimizer.param_groups]

        # send for aggregation
        self.transfer_variable.client_params.remote(params_group, suffix=(self._step,))

        # recv aggregated params
        agg_params_group = self.transfer_variable.aggregated_params.get(idx=0, suffix=(self._step,))

        # set aggregated params
        for agg_group, group in zip(agg_params_group, self.optimizer.param_groups):
            for agg_p, p in zip(agg_group, group["params"]):
                p.data.copy_(torch.Tensor(agg_p))

        # step inc
        self._step += 1


class FedAvgSchedulerAggregator:
    def __init__(self):
        self.transfer_variable = FedAvgSchedulerTransferVariable()
        self._step = 0

    def step(self):
        # recv params for aggregation
        params_groups = self.transfer_variable.client_params.get(suffix=(self._step,))

        # aggregated
        aggregated_params_group = params_groups[0]
        n = len(params_groups)
        for params_group in params_groups[1:]:
            for agg_params, params in zip(aggregated_params_group, params_group):
                for agg_p, p in zip(agg_params, params):
                    agg_p += p / n

        # send aggregated
        self.transfer_variable.aggregated_params.remote(aggregated_params_group, suffix=(self._step,))

        # step inc
        self._step += 1
