from dataclasses import dataclass


@dataclass
class Metric:
    """
    A metric summarizes a list of values for the performance class.
    """

    mean: float
    std: float
