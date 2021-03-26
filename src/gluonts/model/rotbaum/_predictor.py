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

import concurrent.futures
import logging
from itertools import chain
from typing import Iterator, List, Optional

import numpy as np
import pandas as pd

from gluonts.core.component import validated
from gluonts.dataset.common import Dataset
from gluonts.model.forecast import Forecast
from gluonts.model.forecast_generator import log_once
from gluonts.model.predictor import RepresentablePredictor
from gluonts.support.pandas import forecast_start

from ._model import QRF, QRX, QuantileReg
from ._preprocess import Cardinality, PreprocessOnlyLagFeatures

logger = logging.getLogger(__name__)


class RotbaumForecast(Forecast):
    """
    Implements the quantile function in Forecast for TreePredictor,
    as well as a new estimate_dists function for estimating a sampling of the
    conditional distribution of the value of each of the steps in the
    forecast horizon (independently).
    """

    @validated()
    def __init__(
        self,
        models: List,
        featurized_data: List,
        start_date: pd.Timestamp,
        freq,
        prediction_length: int,
    ):
        self.models = models
        self.featurized_data = featurized_data
        self.start_date = start_date
        self.freq = freq
        self.prediction_length = prediction_length
        self.item_id = None
        self.lead_time = None

    def quantile(self, q: float) -> np.array:
        """
        Returns np.array, where the i^th entry is the estimate of the q
        quantile of the conditional distribution of the value of the i^th
        step in the forecast horizon.
        """
        assert 0 <= q <= 1
        return np.array(
            list(
                chain.from_iterable(
                    model.predict(self.featurized_data, q)
                    for model in self.models
                )
            )
        )

    def estimate_dists(self) -> np.array:
        """
        Returns np.array, where the i^th entry is an estimated sampling from
        the conditional distribution of the value of the i^th step in the
        forecast horizon.
        """
        return np.array(
            list(
                chain.from_iterable(
                    model.estimate_dist(self.featurized_data)
                    for model in self.models
                )
            )
        )


class TreePredictor(RepresentablePredictor):
    """
    A predictor that uses a QRX model for each of the steps in the forecast
    horizon. (In other words, there's a total of prediction_length many
    models being trained. In particular, this predictor does not learn a
    multivariate distribution.) The list of these models is saved under
    self.model_list.
    """

    @validated()
    def __init__(
        self,
        freq: str,
        prediction_length: int,
        n_ignore_last: int = 0,
        lead_time: int = 0,
        max_n_datapts: int = 1000000,
        clump_size: int = 100,  # Used only for "QRX" method.
        context_length: Optional[int] = None,
        use_feat_static_real: bool = False,
        use_feat_dynamic_real: bool = False,
        use_feat_dynamic_cat: bool = False,
        cardinality: Cardinality = "auto",
        one_hot_encode: bool = False,
        model_params: Optional[dict] = None,
        max_workers: Optional[int] = None,
        method: str = "QRX",
        quantiles=None,  # Used only for "QuantileRegression" method.
        model=None,
    ) -> None:
        assert method in [
            "QRX",
            "QuantileRegression",
            "QRF",
        ], "method has to be either 'QRX', 'QuantileRegression', or 'QRF'"
        self.method = method
        self.lead_time = lead_time
        self.context_length = (
            context_length if context_length is not None else prediction_length
        )
        self.preprocess_object = PreprocessOnlyLagFeatures(
            self.context_length,
            forecast_horizon=prediction_length,
            stratify_targets=False,
            n_ignore_last=n_ignore_last,
            max_n_datapts=max_n_datapts,
            use_feat_static_real=use_feat_static_real,
            use_feat_dynamic_real=use_feat_dynamic_real,
            use_feat_dynamic_cat=use_feat_dynamic_cat,
            cardinality=cardinality,
            one_hot_encode=one_hot_encode,
        )

        assert (
            context_length is None or context_length > 0
        ), "The value of `context_length` should be > 0"
        assert (
            prediction_length > 0
            or use_feat_dynamic_cat
            or use_feat_dynamic_real
            or use_feat_static_real
            or cardinality
            != "ignore"  # TODO: Figure out how to include 'auto' with no feat_static_cat in this check
        ), (
            "The value of `prediction_length` should be > 0 or there should be features for model training and "
            "prediction "
        )

        self.model_params = model_params if model_params else {}
        self.prediction_length = prediction_length
        self.freq = freq
        self.max_workers = max_workers
        self.clump_size = clump_size
        self.quantiles = quantiles
        self.model = model
        self.model_list = None

        logger.info(
            "If using the Evaluator class with a TreePredictor, set num_workers=0."
        )

    def train(self, training_data):
        assert training_data
        assert self.freq is not None
        if next(iter(training_data))["start"].freq is not None:
            assert self.freq == next(iter(training_data))["start"].freq
        self.preprocess_object.preprocess_from_list(
            ts_list=list(training_data), change_internal_variables=True
        )
        feature_data, target_data = (
            self.preprocess_object.feature_data,
            self.preprocess_object.target_data,
        )
        n_models = self.prediction_length
        logging.info(f"Length of forecast horizon: {n_models}")
        if self.method == "QuantileRegression":
            self.model_list = [
                QuantileReg(params=self.model_params, quantiles=self.quantiles)
                for _ in range(n_models)
            ]
        elif self.method == "QRF":
            self.model_list = [
                QRF(params=self.model_params) for _ in range(n_models)
            ]
        elif self.method == "QRX":
            self.model_list = [
                QRX(
                    xgboost_params=self.model_params,
                    clump_size=self.clump_size,
                    model=self.model,
                )
                for _ in range(n_models)
            ]
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            for n_step, model in enumerate(self.model_list):
                logger.info(
                    f"Training model for step no. {n_step + 1} in the forecast"
                    f" horizon"
                )
                executor.submit(
                    model.fit, feature_data, np.array(target_data)[:, n_step]
                )
        return self

    def predict(
        self, dataset: Dataset, num_samples: Optional[int] = None
    ) -> Iterator[Forecast]:
        """
        Returns a dictionary taking each quantile to a list of floats,
        which are the predictions for that quantile as you run over
        (time_steps, time_series) lexicographically. So: first it would give
        the quantile prediction for the first time step for all time series,
        then the second time step for all time series ˜˜ , and so forth.
        """
        context_length = self.preprocess_object.context_window_size

        if num_samples:
            log_once(
                "Forecast is not sample based. Ignoring parameter `num_samples` from predict method."
            )

        for ts in dataset:
            featurized_data = self.preprocess_object.make_features(
                ts, starting_index=len(ts["target"]) - context_length
            )
            yield RotbaumForecast(
                self.model_list,
                [featurized_data],
                start_date=forecast_start(ts),
                prediction_length=self.prediction_length,
                freq=self.freq,
            )
