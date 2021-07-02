from gluonts.core.component import validated
from gluonts.model.estimator import DummyEstimator

from gluonts.model.trivial.constant import ConstantPredictor
from gluonts.model.trivial.identity import IdentityPredictor
from gluonts.model.trivial.mean import MeanPredictor, MovingAveragePredictor


class ConstantEstimator(DummyEstimator):
    @validated(
        getattr(ConstantPredictor.__init__, "Model")
    )  # Reuse the model Predictor model
    def __init__(self, **kwargs) -> None:
        super().__init__(predictor_cls=ConstantPredictor, **kwargs)


class IdentityEstimator(DummyEstimator):
    @validated(
        getattr(IdentityPredictor.__init__, "Model")
    )  # Reuse the model Predictor model
    def __init__(self, **kwargs) -> None:
        super().__init__(predictor_cls=IdentityPredictor, **kwargs)


class MeanEstimator(DummyEstimator):
    @validated(
        getattr(MeanPredictor.__init__, "Model")
    )  # Reuse the model Predictor model
    def __init__(self, **kwargs) -> None:
        super().__init__(predictor_cls=MeanPredictor, **kwargs)


class MovingAverageEstimator(DummyEstimator):
    @validated(
        getattr(MovingAveragePredictor.__init__, "Model")
    )  # Reuse the model Predictor model
    def __init__(self, **kwargs) -> None:
        super().__init__(predictor_cls=MovingAveragePredictor, **kwargs)
