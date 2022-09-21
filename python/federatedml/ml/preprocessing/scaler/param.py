from federatedml.param.base_param import BaseParam


class FeatureScalerParam(BaseParam):
    def __init__(self, method="standard_scale", scale_col_names=None, scale_col_index=None):
        self.method = method
        self.scale_col_names = scale_col_names
        self.scale_col_indexes = scale_col_index

    def check(self):
        assert self.method in ["standard_scale", "min_max_scale"]
