from typing import List, Optional
from ._base import CandidateGenerator, T


class ReplayCandidateGenerator(CandidateGenerator[T]):
    """
    The replay candidate generator simply returns the model configurations seen during training. If
    candidates are provided, they are returned as is.
    """

    def __init__(self) -> None:
        """
        Initializes a new replay candidate generator.
        """
        self.cache: List[T] = []

    def fit(self, configs: List[T]) -> None:
        self.cache = configs

    def generate(self, candidates: Optional[List[T]] = None) -> List[T]:
        # Assert trained
        assert self.cache, "Replay candidate generator has not been fitted."

        # Return candidates or cache
        return candidates or self.cache
