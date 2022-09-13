from federatedml.param.base_param import BaseParam
from federatedml.util import consts
from .model_io import ModelLoader, ModelSaver
from .context import Context


class Module:
    mode = None

    def __init__(self, params: BaseParam) -> None:
        ...

    def fit(self, ctx: Context, train_data, validate_data=None):
        ...

    def transform(self, ctx: Context, transform_data):
        ...

    def predict(self, ctx: Context, predict_data):
        ...

    @classmethod
    def load_model(cls, ctx: Context, loader: ModelLoader) -> "Module":
        ...

    def save_model(self, ctx: Context, saver: ModelSaver):
        ...


class HeteroModule(Module):
    mode = consts.HETERO


class HomoModule(Module):
    mode = consts.HOMO
