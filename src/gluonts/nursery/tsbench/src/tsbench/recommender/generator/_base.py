from abc import ABC, abstractmethod
from typing import Generic, List, Optional, TypeVar

T = TypeVar("T")


class CandidateGenerator(ABC, Generic[T]):
    """
    A candidate generator provides candidate model configurations to the recommender.
    """

    @abstractmethod
    def fit(self, configs: List[T]) -> None:
        """
        Fits the candidate generator on a list of model configurations seen during training.

        Args:
            configs: The model configurations seen during training.
        """

    @abstractmethod
    def generate(self, candidates: Optional[List[T]] = None) -> List[T]:
        """
        Generates a list of possible model configurations according to the strategy defined by the
        class.

        Args:
            candidates: If provided, every model configuration returned must be a member of this
                set.

        Returns:
            The generated model configurations.
        """
