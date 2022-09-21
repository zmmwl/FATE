from fate_arch.tensor import GUEST, HOST, Arbiter
import numpy as np


class FedAvgAggregatorServer(object):
    def __init__(self, ctx):
        self._ctx = ctx

    def aggregate_loss(self):
        loss = self._ctx.pull(HOST, "loss")
        return np.sum(loss) / len(loss)

    def aggregate_model(self):
        model = self._ctx.pull(HOST, "model")
        return np.sum(model, axis=0)


class FedAvgAggregatorClient(object):
    def __init__(self, ctx):
        self._ctx = ctx

    def aggregate_loss(self, loss):
        self._ctx.push(Arbiter, "loss", loss)

    def aggregate_model(self, model):
        self._ctx.push(Arbiter, "model", model)
