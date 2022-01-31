from typing import List
import numpy as np
import numpy.typing as npt
from tsbench.config import Config
from ._base import Surrogate, T
from ._factory import register_ensemble_surrogate, register_surrogate


@register_surrogate("random")
@register_ensemble_surrogate("random")
class RandomSurrogate(Surrogate[T]):
    """
    The random surrogate simply predicts random performance metrics to act as a baseline.
    """

    num_outputs_: int

    def _fit(self, X: List[Config[T]], y: npt.NDArray[np.float32]) -> None:
        self.num_outputs_ = y.shape[1]

    def _predict(self, X: List[Config[T]]) -> npt.NDArray[np.float32]:
        return np.random.rand(len(X), self.num_outputs_).astype(np.float32)
