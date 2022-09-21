import numpy as np
from federatedml.model_base import ComponentOutput, ModelBase
from federatedml.ml.training.horizontal.homo_lr_param import HomoLRParam
from federatedml.ml.utils.data.dataloader import DataLoader
from federatedml.ml.metrics.loss import SigmoidBinaryCrossEntropyLoss
from fate_arch.tensor import Context
from fate_arch.tensor import HOST, Arbiter
from federatedml.ml.protocol.aggregator import FedAvgAggregatorServer, FedAvgAggregatorClient


class HomoLR(ModelBase):
    def __init__(self):
        super(HomoLR, self).__init__()
        self.model_param = HomoLRParam()
        self._weight = None

    def _init_runtime_parameters(self, cpn_input):
        self.model_param.update(cpn_input.parameters)

    def run(self, cpn_input):
        ctx = Context.from_cpn_input(cpn_input)
        self._init_runtime_parameters(cpn_input)

        train_data = cpn_input["data"]["train_data"]
        validate_data = cpn_input["data"].get("validate_data")

        self.fit(ctx, train_data, validate_data)


class HomoLRClient(HomoLR):
    def __init__(self):
        super(HomoLRClient, self).__init__()
        self._avg_client = None

    def fit(self, ctx, train_data, validate_data=None):
        self._weight = np.array(train_data.shape[1] + 1)
        self._avg_client = FedAvgAggregatorClient(self._ctx)

        batch_generator = DataLoader(train_data, ctx,
                                     mode="homo",
                                     batch_size=self.model_param.batch_size,
                                     random_seed=self.model_param.random_seed,
                                     shuffle=self.model_param.shuffle)

        for it in range(self.model_param.max_iter):
            for batch_id in batch_generator.batch_num:
                X, y = batch_generator.next_batch(with_index=False)
                y_pred = self.forward(X)

                loss = SigmoidBinaryCrossEntropyLoss.compute_loss(y, y_pred, reduction="mean")
                self._avg_client.aggregate(loss)

                # TODO: gradient calculation process should be abstract, the following just an example
                d = SigmoidBinaryCrossEntropyLoss.compute_gradient(y, y_pred, reduction=None)
                weight_grad = (d * X).sum(axis=0) / X.shape[1]
                bias_grad = d.sum(axis=0) / d.shape[1]
                grad = np.hstack((weight_grad, bias_grad))

                weight = self._weight - grad * self.model_param.learning_rate

                self._avg_client.aggregate(weight)
                self._weight = ctx.pull(Arbiter, "aggregated_model")

    def forward(self, X):
        return (X @ self._weight[:-1] + self._weight[1]).sigmoid()

    def predict(self, ctx, test_data):
        predict_prob = self.forward(test_data.values)

        # TODO: should combine index, y, predict_prob, predict_detail


class HomoLRServer(HomoLR):
    def __init__(self):
        super(HomoLRServer, self).__init__()
        self._avg_server = None

    def fit(self, ctx, *args, **kwargs):
        self._avg_server = FedAvgAggregatorServer(self._ctx)
        batch_generator = DataLoader(None, ctx,
                                     mode="homo",
                                     batch_size=self.model_param.batch_size,
                                     random_seed=self.model_param.random_seed,
                                     shuffle=self.model_param.shuffle)

        loss_history = []
        for it in range(self.model_param.max_iter):
            epoch_loss = None
            for batch_id in range(batch_generator.batch_num):
                batch_loss = self._avg_server.aggregate_loss()
                epoch_loss = batch_loss if not epoch_loss else epoch_loss + batch_loss

                model = self._avg_server.aggregate_model()
                ctx.push(HOST, "aggregated_model", model)

            epoch_loss /= batch_generator.batch_num

            loss_history.append(epoch_loss)

    def predict(self):
        # TODO: please note that in some situation, arbiter/server does not need to do predict
        pass
