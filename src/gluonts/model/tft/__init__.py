from ._estimator import TemporalFusionTransformerEstimator

__all__ = ["TemporalFusionTransformerEstimator"]

# fix Sphinx issues, see https://bit.ly/2K2eptM
for item in __all__:
    if hasattr(item, "__module__"):
        setattr(item, "__module__", __name__)
