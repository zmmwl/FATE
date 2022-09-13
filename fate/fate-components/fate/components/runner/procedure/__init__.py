from typing import Optional

from .cross_validation import CrossValidation
from .feature_engineering_fit import FeatureEngineeringFit
from .feature_engineering_transform import FeatureEngineeringTransform
from .predict import Predict
from .procedure import Procedure, Situation
from .skip_run import SkipRun
from .stepwise import Stepwise
from .train_with_validate import TrainWithValidate
from .train_without_validate import TrainWithoutValidate
from .warm_start import WarmStart


def dispatch(situations: Situation) -> Procedure:
    for procedure_cls in [
        SkipRun,
        CrossValidation,
        Stepwise,
        WarmStart,
        TrainWithValidate,
        TrainWithoutValidate,
        Predict,
        FeatureEngineeringFit,
        FeatureEngineeringTransform,
    ]:
        procedure = procedure_cls(situations)
        if procedure.is_activate:
            return procedure
    raise RuntimeError(f"dispatch nothing, situations: {Situation}")


__all__ = [
    "CrossValidation",
    "FeatureEngineeringTransform",
    "FeatureEngineeringFit",
    "Predict",
    "Stepwise",
    "TrainWithValidate",
    "TrainWithoutValidate",
    "WarmStart",
    "SkipRun",
    "Situation",
    "dispatch",
]
