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
import logging
from typing import Callable, Optional, List, Tuple
import pandas as pd
from autogluon.tabular import TabularPredictor as AutogluonTabularPredictor

from gluonts.core.component import validated
from gluonts.dataset.common import Dataset
from gluonts.dataset.util import to_pandas
from gluonts.model.estimator import Estimator
from gluonts.time_feature import (
    TimeFeature,
    get_lags_for_frequency,
    time_features_from_frequency_str,
)

from .predictor import (
    TabularPredictor,
    mean_abs_scaling,
    get_features_dataframe,
)

logger = logging.getLogger(__name__)


class TabularEstimator(Estimator):
    """
    An estimator that trains an Autogluon Tabular model for time series
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
    lag_indices
        List of indices of the lagged observations to use as features. If
        None, this will be set automatically based on the frequency.
    time_features
        List of time features to be used. If None, this will be set automatically
        based on the frequency.
    scaling
        Function to be used to scale time series. This should take a pd.Series object
        as input, and return a scaled pd.Series and the scale (float). By default,
        this divides a series by the mean of its absolute value.
    batch_size
        Batch size of the resulting predictor; this is just used at prediction
        time, and does not affect training in any way.
    disable_auto_regression
        Whether to forcefully disable auto-regression in the model. If ``True``,
        this will remove any lag index which is smaller than ``prediction_length``.
        This will make predictions more efficient, but may impact their accuracy.
    quantiles_to_predict
        Whether to forecast in quantile way. If assigned with quantile values,
        this will train model using quantile prediction model. If None, then the model
        will be trained in a regular way.
    """

    @validated()
    def __init__(
        self,
        freq: str,
        prediction_length: int,
        lag_indices: Optional[List[int]] = None,
        time_features: Optional[List[TimeFeature]] = None,
        scaling: Callable[
            [pd.Series], Tuple[pd.Series, float]
        ] = mean_abs_scaling,
        batch_size: Optional[int] = 32,
        disable_auto_regression: bool = False,
        last_k_for_val: Optional[int] = None,
        quantiles_to_predict: Optional[List[float]] = None,
        eval_metric: str = "mean_absolute_error",
        **kwargs,
    ) -> None:
        super().__init__()

        self.prediction_length = prediction_length
        self.lag_indices = (
            lag_indices
            if lag_indices is not None
            else get_lags_for_frequency(freq)
        )
        self.time_features = (
            time_features
            if time_features is not None
            else time_features_from_frequency_str(freq)
        )
        self.batch_size = batch_size
        self.disable_auto_regression = disable_auto_regression
        self.scaling = scaling
        self.last_k_for_val = last_k_for_val
        self.eval_metric = eval_metric
        self.quantiles_to_predict = quantiles_to_predict

        if self.disable_auto_regression:
            self.lag_indices = [
                lag_idx
                for lag_idx in self.lag_indices
                if lag_idx >= self.prediction_length
            ]

        default_kwargs = {
            "time_limit": 60,
            # "excluded_model_types": ["KNN", "XT", "RF"],
            "presets": [
                "high_quality_fast_inference_only_refit",
                "optimize_for_deployment",
            ],
            "auto_stack": True,
        }
        self.kwargs = {**default_kwargs, **kwargs}

    def train(
        self,
        training_data: Dataset,
        validation_data: Optional[Dataset] = None,
    ) -> TabularPredictor:

        kwargs_override = {}

        dfs = [
            get_features_dataframe(
                series=self.scaling(to_pandas(entry))[0],
                time_features=self.time_features,
                lag_indices=self.lag_indices,
            )
            for entry in training_data
        ]

        if validation_data is not None or self.last_k_for_val is not None:
            kwargs_override["auto_stack"] = False
            logger.warning(
                "Auto Stacking is turned off as validation dataset is provided"
                " before input into Tabular Predictor."
            )

        if validation_data is not None:
            logger.log(20, "Validation dataset is directly provided.")
            validation_dfs = [
                get_features_dataframe(
                    series=self.scaling(to_pandas(entry))[0],
                    time_features=self.time_features,
                    lag_indices=self.lag_indices,
                )
                for entry in validation_data
            ]
            train_df = pd.concat(dfs)
            val_df = pd.concat(validation_dfs)
        elif self.last_k_for_val is not None:
            logger.log(
                20,
                "last_k_for_val is provided, choosing last"
                f" {self.last_k_for_val} of each time series as validation"
                " set.",
            )
            train_dfs = [
                tmp_df.iloc[: -self.last_k_for_val, :] for tmp_df in dfs
            ]
            validation_dfs = [
                tmp_df.iloc[-self.last_k_for_val :, :] for tmp_df in dfs
            ]
            train_df = pd.concat(train_dfs)
            val_df = pd.concat(validation_dfs)
        else:
            logger.log(
                20,
                "No validation dataset is provided, will let TabularPredictor"
                " do the splitting automatically,Note that this might break"
                " the time order of time series data.",
            )
            train_df = pd.concat(dfs)
            val_df = None

        if self.quantiles_to_predict is not None:
            ag_model = AutogluonTabularPredictor(
                label="target",
                problem_type="quantile",
                quantile_levels=self.quantiles_to_predict,
            ).fit(
                train_df,
                tuning_data=val_df,
                **{**self.kwargs, **kwargs_override},
            )
        else:
            ag_model = AutogluonTabularPredictor(
                label="target",
                problem_type="regression",
                eval_metric=self.eval_metric,
            ).fit(
                train_df,
                tuning_data=val_df,
                **{**self.kwargs, **kwargs_override},
            )

        return TabularPredictor(
            ag_model=ag_model,
            prediction_length=self.prediction_length,
            time_features=self.time_features,
            lag_indices=self.lag_indices,
            scaling=self.scaling,
            batch_size=self.batch_size,
            quantiles_to_predict=self.quantiles_to_predict,
        )
