from dataclasses import dataclass
from typing import Generic, TypeVar
from tsbench.evaluations.metrics import Performance

T = TypeVar("T")


@dataclass
class Recommendation(Generic[T]):
    """
    A recommendation provides a configuration along with its expected performance.
    """

    config: T
    performance: Performance
