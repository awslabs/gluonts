# First-party imports
from gluonts.core.component import validated
from gluonts.model.estimator import DummyEstimator

# Relative imports
from ._predictor import SeasonalNaivePredictor


class SeasonalNaiveEstimator(DummyEstimator):
    """
    An estimator that, upon `train`, simply returns a pre-constructed.
    `SeasonalNaivePredictor`.

    Parameters
    ----------
    kwargs
        Arguments to pass to the `SeasonalNaivePredictor` constructor.
    """

    @validated(
        getattr(SeasonalNaivePredictor.__init__, 'Model')
    )  # Reuse the model Predictor model
    def __init__(self, **kwargs) -> None:
        super().__init__(predictor_cls=SeasonalNaivePredictor, **kwargs)
