from typing import Optional, Protocol, List

from .context import Context
from .data_io import Dataframe
from .model_io import ModelsLoader, ModelsSaver
from .param import Param


class Module(Protocol):
    mode: str

    def __init__(self, params: Param) -> None:
        ...

    def fit(
        self,
        ctx: Context,
        train_data: Dataframe,
        validate_data: Optional[Dataframe] = None,
    ) -> None:
        ...

    def transform(self, ctx: Context, transform_data: Dataframe) -> List[Dataframe]:
        ...

    def predict(self, ctx: Context, predict_data: Dataframe) -> Dataframe:
        ...

    @classmethod
    def load_model(cls, ctx: Context, loader: ModelsLoader) -> "Module":
        ...

    def save_model(self, ctx: Context, saver: ModelsSaver) -> None:
        ...
