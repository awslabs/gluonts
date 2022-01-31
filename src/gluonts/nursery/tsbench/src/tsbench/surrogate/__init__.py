from ._base import Surrogate
from ._factory import (
    create_ensemble_surrogate,
    create_surrogate,
    ENSEMBLE_SURROGATE_REGISTRY,
    SURROGATE_REGISTRY,
)
from .autogluon import AutoGluonSurrogate
from .deepset import DeepSetSurrogate
from .mlp import MLPSurrogate
from .nonparametric import NonparametricSurrogate
from .random import RandomSurrogate
from .random_forest import RandomForestSurrogate
from .xgboost import XGBoostSurrogate

__all__ = [
    "AutoGluonSurrogate",
    "DeepSetSurrogate",
    "ENSEMBLE_SURROGATE_REGISTRY",
    "MLPSurrogate",
    "NonparametricSurrogate",
    "RandomForestSurrogate",
    "RandomSurrogate",
    "SURROGATE_REGISTRY",
    "Surrogate",
    "XGBoostSurrogate",
    "create_ensemble_surrogate",
    "create_surrogate",
]

# We need to set some parallelism flags to ensure that PyTorch behaves well on beastier machines
import torch  # pylint: disable=wrong-import-order

torch.set_num_threads(4)
torch.set_num_interop_threads(4)
