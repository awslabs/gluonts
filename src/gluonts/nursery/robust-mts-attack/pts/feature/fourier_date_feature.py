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

from typing import List

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset

from gluonts.core.component import validated
from gluonts.time_feature import TimeFeature, norm_freq_str


class FourierDateFeatures(TimeFeature):
    @validated()
    def __init__(self, freq: str) -> None:
        super().__init__()
        # reocurring freq
        freqs = [
            "month",
            "day",
            "hour",
            "minute",
            "weekofyear",
            "weekday",
            "dayofweek",
            "dayofyear",
            "daysinmonth",
        ]

        assert freq in freqs
        self.freq = freq

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        values = getattr(index, self.freq)
        num_values = max(values) + 1
        steps = [x * 2.0 * np.pi / num_values for x in values]
        return np.vstack([np.cos(steps), np.sin(steps)])


def fourier_time_features_from_frequency(freq_str: str) -> List[TimeFeature]:
    offset = to_offset(freq_str)
    granularity = norm_freq_str(offset.name)

    features = {
        "M": ["weekofyear"],
        "W": ["daysinmonth", "weekofyear"],
        "D": ["dayofweek"],
        "B": ["dayofweek", "dayofyear"],
        "H": ["hour", "dayofweek"],
        "min": ["minute", "hour", "dayofweek"],
        "T": ["minute", "hour", "dayofweek"],
    }

    assert granularity in features, f"freq {granularity} not supported"

    feature_classes: List[TimeFeature] = [
        FourierDateFeatures(freq=freq) for freq in features[granularity]
    ]
    return feature_classes
