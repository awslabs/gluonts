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

from typing import Callable, Optional, List, Tuple

import pandas as pd
from autogluon import TabularPrediction as task

from gluonts.dataset.common import Dataset
from gluonts.dataset.util import to_pandas
from gluonts.model.estimator import Estimator
from gluonts.time_feature import get_lags_for_frequency

from .predictor import (
    TabularPredictor,
    mean_abs_scaling,
    get_features_dataframe,
)


class TabularEstimator(Estimator):
    """An estimator that trains an Autogluon Tabular model for time series
    forecasting.

    Additional keyword arguments to the constructor, other than the ones documented
    below, will be passed on to Autogluon Tabular's ``fit`` method used for training
    the model.

    Parameters
    ----------
    freq
        Frequency of the data to handle
    prediction_length
        Prediction length
    lags
        List of indices of the lagged observations to use as features. If
        None, this will be set automatically based on the frequency.
    scaling
        Function to be used to scale time series. This should take a pd.Series object
        as input, and return a scaled pd.Series and the scale (float). By default,
        this divides a series by the mean of its absolute value.
    batch_size
        Batch size of the resulting predictor; this is just used at prediction
        time, and does not affect training in any way.
    disable_auto_regression
        Weather to forecefully disable auto-regression in the model. If ``True``,
        this will remove any lag index which is smaller than ``prediction_length``.
        This will make predictions more efficient, but may impact their accuracy.
    """

    def __init__(
        self,
        freq: str,
        prediction_length: int,
        lags: Optional[List[int]] = None,
        scaling: Callable[
            [pd.Series], Tuple[pd.Series, float]
        ] = mean_abs_scaling,
        batch_size: Optional[int] = 32,
        disable_auto_regression: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        self.task = task
        self.freq = freq
        self.prediction_length = prediction_length
        self.lags = (
            lags if lags is not None else (get_lags_for_frequency(self.freq))
        )
        self.batch_size = batch_size
        self.disable_auto_regression = disable_auto_regression
        self.scaling = scaling

        if self.disable_auto_regression:
            self.lags = [l for l in self.lags if l >= self.prediction_length]

        default_kwargs = {
            "eval_metric": "mean_absolute_error",
            "excluded_model_types": ["KNN", "XT", "RF"],
            "presets": [
                "high_quality_fast_inference_only_refit",
                "optimize_for_deployment",
            ],
        }
        self.kwargs = {**default_kwargs, **kwargs}

    def train(self, training_data: Dataset) -> TabularPredictor:
        dfs = [
            get_features_dataframe(
                series=self.scaling(to_pandas(entry))[0],
                lags=self.lags,
            )
            for entry in training_data
        ]
        df = pd.concat(dfs)
        ag_model = self.task.fit(
            df, label="target", problem_type="regression", **self.kwargs
        )
        return TabularPredictor(
            ag_model=ag_model,
            freq=self.freq,
            prediction_length=self.prediction_length,
            lags=self.lags,
            scaling=self.scaling,
            batch_size=self.batch_size,
        )
