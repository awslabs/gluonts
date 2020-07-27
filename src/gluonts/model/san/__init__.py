from ._estimator import SelfAttentionEstimator
from ._engine import Trainer as SelfAttentionTrainer

__all__ = [
    "SelfAttentionEstimator",
    "SelfAttentionTrainer",
]

# fix Sphinx issues, see https://bit.ly/2K2eptM
for item in __all__:
    if hasattr(item, "__module__"):
        setattr(item, "__module__", __name__)
