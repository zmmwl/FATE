import numpy as np
from federatedml.model_base import ComponentOutput, ModelBase
from federatedml.ml.training.vertical.hetero_sshe_lr_param import HeteroSSHELRParam
from federatedml.ml.utils.data.dataloader import DataLoader
from federatedml.ml.metrics.loss import SigmoidBinaryCrossEntropyLoss
from fate_arch.tensor import Context, CipherKind
from fate_arch.tensor import GUEST, HOST, Arbiter
from federatedml.ml.protocol.aggregator import FedAvgAggregatorServer, FedAvgAggregatorClient


class HeteroSSHELR(ModelBase):
    def __init__(self):
        super(HeteroSSHELR, self).__init__()
        self.model_param = HeteroSSHELRParam()
        self._ctx = None
        self._weight = None
        self._phe_pk = None
        self._phe_sk = None

    def _init_runtime_parameters(self, cpn_input):
        self.model_param.update(cpn_input.parameters)

    def run(self, cpn_input):
        ctx = Context.from_cpn_input(cpn_input)
        self._init_runtime_parameters(cpn_input)

        train_data = cpn_input["data"]["train_data"]
        validate_data = cpn_input["data"].get("validate_data")

        self._phe_pk, self._phe_sk = ctx.keygen(CipherKind.PHE, 1024)
        self.fit(ctx, train_data, validate_data)

    @property
    def fit_intercept(self):
        return self._ctx.is_guest

    def share_model(self, ctx):
        raise NotImplementedError


class HeteroSSHELRGuest(HeteroSSHELR):
    def __init__(self):
        super(HeteroSSHELRGuest, self).__init__()

    def share_model(self, ctx):
        weight = self._weight[:-1]
        wb = self._ctx.ShareFPTensor.from_source("wb",
                                                 ctx,
                                                 [weight, HOST],
                                                 base=self.model_param.base,
                                                 precision=self.model_param.precision)
        wa = self._ctx.FPTensor.from_source("wa",
                                            ctx)

        return wb, wa

    def fit(self, ctx, train_data, validate_data=None):
        self._ctx = ctx
        self._weight = np.array(train_data.shape[1] + 1)

        batch_generator = DataLoader(train_data, ctx,
                                     mode="homo",
                                     batch_size=self.model_param.batch_size,
                                     random_seed=self.model_param.random_seed,
                                     shuffle=self.model_param.shuffle)

        w_self, w_other = self.share_model(ctx)

        for it in range(self.model_param.max_iter):
            for batch_id in batch_generator.batch_num:
                X, y = batch_generator.next_batch(with_index=False)
                # To Be Decided: do we need to transform general tensor to FPTenser explicitly

                y_pred = self.forward(w_self, w_other, X)

    def forward(self, w_self, w_other, X):
        # begin to cal wx_guest
        zb_local = X @ w_self # local dot

        za_phe_tensor = self._ctx.pull("za_phe_tensor",
                                       HOST).unwrap_phe_tensor()

        za_dot = X.share_dot(za_phe_tensor)
        za_dot_share = self._ctx.rand_tensor(za_dot.shape)

        self._ctx.push("za_dot_share", HOST, za_dot - za_dot_share)

        zb_phe_tensor = self._phe_pk.encrypt(w_self)
        self._ctx.push("zb_phe_tensor", HOST, zb_phe_tensor)

        zb_dot_share = self._ctx.pull("zb_dot_share", HOST).unwrap_phe_tensor()
        zb_dot_share = self._phe_sk.decrypt(zb_dot_share)

        wx_guest = zb_local + za_dot_share + zb_dot_share
        # finish to cal wx_guest

        wx_host_phe = self._ctx.pull("wx_host_phe", HOST).unwrap_phe_tensor()

        sigmoid_z = (wx_guest + wx_host_phe) * 0.25 + 0.5
        sigmoid_z_share = self._ctx.rand_tensor(sigmoid_z.shape)
        self._ctx.push("sigmoid_z", HOST, sigmoid_z - sigmoid_z_share)

        return sigmoid_z


class HeteroSSHELRHost(HeteroSSHELR):
    def __init__(self):
        super(HeteroSSHELRHost, self).__init__()
        self._avg_server = None

    def share_model(self, ctx):
        weight = self._weight[:-1]
        wa = self._ctx.SHareFPTensor.from_source("wa",
                                                 ctx,
                                                 [weight, GUEST],
                                                 base=self.model_param.base,
                                                 precision=self.model_param.precision)
        wb = self._ctx.ShareFPTensor.from_source("wb",
                                                 ctx)

        return wa, wb

    def fit(self, ctx, train_data, validate_data=None):
        self._ctx = ctx
        self._weight = np.array(train_data.shape[1])

        batch_generator = DataLoader(None, ctx,
                                     mode="homo",
                                     batch_size=self.model_param.batch_size,
                                     random_seed=self.model_param.random_seed,
                                     shuffle=self.model_param.shuffle)

        w_self, w_other = self.share_model(ctx)

        for it in range(self.model_param.max_iter):
            for batch_id in range(batch_generator.batch_num):
                X, y = batch_generator.next_batch(with_index=False)
                # To Be Decided: do we need to transform general tensor to FPTenser explicitly

                self.forward(w_self, w_other, X)

    def forward(self, w_self, w_other, X):
        zb_local = X @ w_self  # local dot

        za_phe_tensor = self._phe_pk.encrypt(w_self)
        self._ctx.push("za_phe_tensor", GUEST, za_phe_tensor)
        za_dot_share = self._ctx.pull("za_dot_share", HOST).unwrap_phe_tensor()
        za_dot_share = self._phe_sk.decrypt(za_dot_share)

        zb_phe_tensor = self._ctx.pull("zb_phe_tensor", GUEST).unwrap_phe_tensor()

        zb_dot = X.share_dot(zb_phe_tensor)
        zb_dot_share = self._ctx.rand_tensor(zb_dot.shape)

        self._ctx.push("zb_dot_share", HOST, zb_dot - zb_dot_share)

        wx_host = zb_local + za_dot_share + zb_dot_share

        wx_host_phe = self._phe_pk.encrypt(wx_host)
        self._cts.push("wx_host_phe", GUEST, wx_host_phe)

        sigmoid_z = self._ctx.pull("sigmoid_z", GUEST).unwrap_phe_tensor()
        sigmoid_z = self._phe_sk.decrypt(sigmoid_z)

        return sigmoid_z


