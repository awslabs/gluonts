# First-party imports
from gluonts.core.component import validated
from gluonts.model.estimator import DummyEstimator

# Relative imports
from ._predictor import NPTSPredictor


class NPTSEstimator(DummyEstimator):
    @validated(
        getattr(NPTSPredictor.__init__, 'Model')
    )  # Reuse the model Predictor model
    def __init__(self, **kwargs) -> None:
        super().__init__(predictor_cls=NPTSPredictor, **kwargs)
