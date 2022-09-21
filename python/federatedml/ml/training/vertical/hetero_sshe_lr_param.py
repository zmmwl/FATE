from federatedml.param.base_param import BaseParam


class HeteroSSHELRParam(BaseParam):
    def __init__(self, penalty="L2", tol=1e-4, alpha=1.0, optimizer="rmsprop",
                 batch_size=-1, learning_rate=0.01, max_iter=100):
        self.penalty = penalty
        self.tol = tol
        self.alpha = alpha
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_iter = max_iter

    def check(self):
        assert self.method in ["standard_scale", "min_max_scale"]
