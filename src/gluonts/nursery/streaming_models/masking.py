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

from typing import Any, Callable, Tuple

import numpy as np
import pandas as pd

from gluonts.core.component import validated
from gluonts.mx.model.forecast import DistributionForecast
from gluonts.transform.feature import RollingMeanValueImputation

from .predictor import StreamState
from .scoring_functions import ScoringFunction, TailProbability


class MaskingStrategy:
    anomaly_history_size: int

    def mask(
        self,
        masking_state: StreamState,
        forecast: DistributionForecast,
        values: pd.Series,
    ) -> Tuple[StreamState, pd.Series, pd.Series]:
        raise NotImplementedError()


class MissingValueMaskingStrategy(MaskingStrategy):
    """
    A simple masking strategy that masks missing values with samples
    from the predictive distribution.
    """

    def mask(
        self,
        masking_state: StreamState,
        forecast: DistributionForecast,
        values: pd.Series,
    ) -> Tuple[StreamState, pd.Series, pd.Series]:
        maybe_masked = []
        is_masked = []

        samples = forecast.distribution.sample().asnumpy()
        for value, sample in zip(values.target, samples):

            is_masked.append(np.isnan(float(value)))
            maybe_masked.append(sample if np.isnan(float(value)) else value)

        start = values.start
        return (
            masking_state,
            pd.Series(maybe_masked, index=values.index),
            pd.Series(is_masked, index=values.index),
        )


class SimpleMaskingStrategy(MaskingStrategy):
    """
    A simple masking strategy that masks values with scores above the threshold,
    except when the fraction of anomalous points in the recent history is above a
    certain value.

    Parameters
    ----------
    score_threshold
        Scores above this values will be masked
    accept_anomaly_fraction
        Maximum fraction of points that can be anomalous in the history before
        we stop masking.
    anomaly_history_size
        Window size to consider when computing the fraction of anomalous points.
    num_points_to_accept
        Number of points to accept independent of the score after the decision is
        made to accept a change.
    mask_missing_values
        Boolean indicating if missing values should be masked with the sample value.
    """

    @validated()
    def __init__(
        self,
        score_threshold: float = 2.0,
        accept_anomaly_fraction: float = 0.5,
        anomaly_history_size: int = 60 * 3,
        num_points_to_accept: int = 60 * 6,
        mask_missing_values: bool = True,
        scoring_function: ScoringFunction = TailProbability(),
    ):
        assert num_points_to_accept > anomaly_history_size
        self.score_threshold = score_threshold
        self.max_frac_anomalies = accept_anomaly_fraction
        self.anomaly_history_size = anomaly_history_size
        self.num_points_to_accept = num_points_to_accept
        self.mask_missing_values = mask_missing_values
        self.scoring_function = scoring_function
        self.should_accept_imputation = RollingMeanValueImputation(
            window_size=max(1, int(self.anomaly_history_size // 2))
        )

    def mask(
        self,
        masking_state: StreamState,
        forecast: DistributionForecast,
        values: pd.Series,
    ) -> Tuple[StreamState, pd.Series, pd.Series]:
        scores_buffer = self.scoring_function((forecast, values))
        samples = forecast.distribution.sample().asnumpy()

        prev_anomalies = masking_state.get("prev_anomalies", [])
        should_accept = masking_state.get(
            "should_accept", self.num_points_to_accept
        )

        score_array = np.array(scores_buffer.target)
        anomalies = score_array > self.score_threshold

        maybe_masked = []
        is_masked = []

        assert len(samples) == len(anomalies) == len(values)
        for value, is_anomaly, sample in zip(
            values.target, anomalies, samples
        ):
            if (
                np.mean(prev_anomalies[-self.anomaly_history_size :])
                > self.max_frac_anomalies
            ):
                if not should_accept:
                    should_accept = self.num_points_to_accept

            if should_accept > 0:
                maybe_masked.append(
                    self.should_accept_imputation(np.array([value]))[0]
                    if self.mask_missing_values and np.isnan(float(value))
                    else value
                )
                is_masked.append(False)
                prev_anomalies.append(False)
                if not np.isnan(value):
                    should_accept -= 1
                continue

            prev_anomalies.append(is_anomaly)
            is_masked.append(is_anomaly)
            maybe_masked.append(
                sample
                if is_anomaly
                or (self.mask_missing_values and np.isnan(float(value)))
                else value
            )

        masking_state["prev_anomalies"] = prev_anomalies[
            -self.anomaly_history_size :
        ]
        masking_state["should_accept"] = should_accept
        start = values.start
        return (
            masking_state,
            pd.Series(maybe_masked, index=values.index),
            pd.Series(is_masked, index=values.index),
        )
