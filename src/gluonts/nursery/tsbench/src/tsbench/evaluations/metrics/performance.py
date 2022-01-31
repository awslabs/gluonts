from __future__ import annotations
from dataclasses import dataclass
from typing import cast, Dict, List, Union
import numpy as np
import pandas as pd
from .metric import Metric


@dataclass
class Performance:
    """
    The performance class encapsulates the metrics that are recorded for configurations.
    """

    training_time: Metric
    latency: Metric
    num_model_parameters: Metric
    num_gradient_updates: Metric

    ncrps: Metric
    mase: Metric
    smape: Metric
    nrmse: Metric
    nd: Metric

    @classmethod
    def from_dict(cls, metrics: Dict[str, Union[float, int]]) -> Performance:
        """
        Initializes a new performance object from the given 1D dictionary. Metrics are expected to
        be provided via `<metric>_mean` and `<metric>_std` keys.
        """
        kwargs = {m: Metric(metrics[f"{m}_mean"], metrics[f"{m}_std"]) for m in cls.metrics()}
        return Performance(**kwargs)  # type: ignore

    @classmethod
    def metrics(cls) -> List[str]:
        """
        Returns the list of metrics that are exposed by the performance class.
        """
        # pylint: disable=no-member
        return list(cls.__dataclass_fields__.keys())  # type: ignore

    @classmethod
    def to_dataframe(cls, performances: List[Performance], std: bool = True) -> pd.DataFrame:
        """
        Returns a data frame representing the provided performances.
        """
        fields = sorted(Performance.__dataclass_fields__.keys())  # pylint: disable=no-member
        result = np.empty((len(performances), 18 if std else 9))

        offset = 2 if std else 1
        for i, performance in enumerate(performances):
            for j, field in enumerate(fields):
                result[i, j * offset] = cast(Metric, getattr(performance, field)).mean
                if std:
                    result[i, j * offset + 1] = cast(Metric, getattr(performance, field)).std

        return pd.DataFrame(
            result,
            columns=[
                f
                for field in fields
                for f in ([f"{field}_mean", f"{field}_std"] if std else [f"{field}_mean"])
            ],
        )
