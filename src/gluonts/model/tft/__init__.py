from ._estimator import TemporalFusionTransformerEstimator
from ._engine import Trainer as TemporalFusionTransformerTrainer

__all__ = [
    "TemporalFusionTransformerEstimator",
    "TemporalFusionTransformerTrainer",
]

# fix Sphinx issues, see https://bit.ly/2K2eptM
for item in __all__:
    if hasattr(item, "__module__"):
        setattr(item, "__module__", __name__)
