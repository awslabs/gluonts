from abc import ABC, abstractmethod
from typing import Generic, TypeVar
from tsbench.config import Config, EnsembleConfig, ModelConfig
from ._evaluations import Evaluations, Performance

T = TypeVar("T", ModelConfig, EnsembleConfig)


class Tracker(ABC, Generic[T]):
    """
    Base class for trackers that draw from local data.
    """

    @abstractmethod
    def get_evaluations(self) -> Evaluations[T]:
        """
        Returns all evaluations from the jobs associated with the tracker.
        """

    @abstractmethod
    def get_performance(self, config: Config[T]) -> Performance:
        """
        Returns the performance metrics for the provided configuration.

        Args:
            config: The configuration object of the type of configuration that the tracker manages.

        Returns:
            The performance metrics.
        """
