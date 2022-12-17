# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

from __future__ import annotations
from dataclasses import dataclass
from typing import cast, Dict, List, Union
import numpy as np
import pandas as pd
from .metric import Metric


@dataclass
class Performance:
    """
    The performance class encapsulates the metrics that are recorded for
    configurations.
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
    def from_dict(cls, metrics: dict[str, float | int]) -> Performance:
        """
        Initializes a new performance object from the given 1D dictionary.

        Metrics are expected to be provided via `<metric>_mean` and
        `<metric>_std` keys.
        """
        kwargs = {
            m: Metric(metrics[f"{m}_mean"], metrics[f"{m}_std"])
            for m in cls.metrics()
        }
        return Performance(**kwargs)  # type: ignore

    @classmethod
    def metrics(cls) -> list[str]:
        """
        Returns the list of metrics that are exposed by the performance class.
        """
        # pylint: disable=no-member
        return list(cls.__dataclass_fields__.keys())  # type: ignore

    @classmethod
    def to_dataframe(
        cls, performances: list[Performance], std: bool = True
    ) -> pd.DataFrame:
        """
        Returns a data frame representing the provided performances.
        """
        fields = sorted(
            Performance.__dataclass_fields__.keys()
        )  # pylint: disable=no-member
        result = np.empty((len(performances), 18 if std else 9))

        offset = 2 if std else 1
        for i, performance in enumerate(performances):
            for j, field in enumerate(fields):
                result[i, j * offset] = cast(
                    Metric, getattr(performance, field)
                ).mean
                if std:
                    result[i, j * offset + 1] = cast(
                        Metric, getattr(performance, field)
                    ).std

        return pd.DataFrame(
            result,
            columns=[
                f
                for field in fields
                for f in (
                    [f"{field}_mean", f"{field}_std"]
                    if std
                    else [f"{field}_mean"]
                )
            ],
        )
