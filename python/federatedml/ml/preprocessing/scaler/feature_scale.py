from federatedml.model_base import ComponentOutput, ModelBase
from federatedml.ml.preprocessing.scaler.param import FeatureScalerParam
from fate_arch.tensor import Context


class MinMaxScaler(object):
    def __init__(self, scale_col_names=None):
        self._scale_col_names = scale_col_names
        self._data_max = None
        self._data_min = None

    def fit(self, ctx, data):
        self._data_max = data.max()
        self._data_min = data.min()

    def transform(self, ctx, data):
        return (data - self._data_min) / (self._data_max - self._data_min)


class StandardScaler(object):
    def __init__(self, scale_col_names=None):
        self._scale_col_names = scale_col_names
        self._data_mean = None
        self._data_std = None

    def fit(self, ctx, data):
        scale_data = data

        if set(self._scale_col_names) != set(data.columns):
            scale_data = data[self._scale_col_names]

        self._data_mean = scale_data.mean()
        self._data_std = scale_data.std()

    def transform(self, ctx, data):
        return (data - self._data_mean) / self._data_std


class FeatureScaler(ModelBase):
    def __init__(self):
        super(FeatureScaler, self).__init__()
        self.model_param = FeatureScalerParam()
        self._scale_runner = None
        self._scale_column_names = None

    def _init_runtime_parameters(self, cpn_input):
        self.model_param.update(cpn_input.parameters)

    def run(self, cpn_input):
        ctx = Context.from_cpn_input(cpn_input)
        self._init_runtime_parameters(cpn_input)

        data = cpn_input["data"]["data"]
        self.fit(ctx, data)

    def fit(self, ctx, data):
        self._scale_column_names = list(set(self.model_param.scale_coloumn_names)
                                        | set(data.columns[self.model_param.select_column_indexes]))

        if self.model_param.method == "standard_scale":
            self._scale_runner = StandardScaler(self._scale_column_names)
        elif self.model_param.method == "min_max_scale":
            self._scale_runner = MinMaxScaler(self._scale_column_names)

        scale_data = data
        if set(self._scale_col_names) != set(data.columns):
            scale_data = data[self._scale_col_names]

        self._scale_runner.fit(ctx, scale_data)
        transform_df = self.transform(ctx, data)
        return self._post_process(transform_df, data)

    def transform(self, ctx, data):
        scale_data = data
        if set(self._scale_col_names) != set(data.columns):
            scale_data = data[self._scale_col_names]

        transform_df = self._scale_runner.transform(ctx, scale_data)
        return self._post_process(transform_df, data)

    def _post_process(self, transform_df, data):
        data[self._scale_column_names] = transform_df
        return data
