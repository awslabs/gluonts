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

# Standard library imports
from enum import Enum
from typing import Iterator, List, Optional, Tuple, Union

# Third-party imports
import numpy as np
import pandas as pd
from itertools import chain
import concurrent.futures

# First-party imports
from gluonts.core.component import validated
from gluonts.dataset.common import Dataset
from gluonts.model.forecast import Forecast, SampleForecast
from gluonts.model.forecast_generator import log_once
from gluonts.model.predictor import RepresentablePredictor
from gluonts.support.pandas import forecast_start

# Relative imports
from ._preprocess import PreprocessOnlyLagFeatures
from ._model import QRX


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
        prediction_length: int
    ):
        self.models = models
        self.featurized_data = featurized_data
        self.start_date = start_date
        self.freq = freq
        self.prediction_length = prediction_length
        self.item_id = None
        self.lead_time = None

    @validated()
    def quantile(self, q: float) -> np.array:
        """
        Returns np.array, where the i^th entry is the estimate of the q
        quantile of the conditional distribution of the value of the i^th
        step in the forecast horizon.
        """
        assert 0 <= q <= 1
        return np.array(
            list(
                chain(
                    *[
                        model.predict(self.featurized_data, q)
                        for model in self.models
                    ]
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
                chain(
                    *[
                        model.estimate_dist(self.featurized_data)
                        for model in self.models
                    ]
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
        context_length: Optional[int],
        prediction_length: Optional[int],
        n_ignore_last: int = 0,
        lead_time: int = 0,
        max_workers: int = 10,
        max_n_datapts: int = 400000,
        model_params=None,
        freq=None,
        use_feat_static_real=False,
        use_feat_static_cat=False,
    ) -> None:
        self.lead_time = lead_time
        self.preprocess_object = PreprocessOnlyLagFeatures(
            context_length,
            forecast_horizon=prediction_length,
            stratify_targets=False,
            n_ignore_last=n_ignore_last,
            max_n_datapts=max_n_datapts,
            use_feat_static_real=use_feat_static_real,
            use_feat_static_cat=use_feat_static_cat
        )
        self.context_length = context_length
        self.model_params = model_params
        self.prediction_length = prediction_length
        self.freq = freq
        self.max_workers = max_workers
        self.model_list = None

    @validated()
    def __call__(self, training_data):
        assert training_data
        if self.freq is not None:
            if next(iter(trainind_data))['start'].freq is None:
                assert self.freq == next(iter(training_data))["start"].freq
        else:
            self.freq = next(iter(training_data))["start"].freq
        self.preprocess_object.preprocess_from_list(
            ts_list=list(training_data), change_internal_variables=True
        )
        feature_data, target_data = (
            self.preprocess_object.feature_data,
            self.preprocess_object.target_data,
        )
        n_models = self.prediction_length
        print(f"Length of forecast horizon: {n_models}")
        self.model_list = [QRX() for _ in range(n_models)]
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_workers) as executor:
            for n_step, model in enumerate(self.model_list):
                print(
                    f"Training model for step no. {n_step + 1} in the forecast"
                    f" horizon"
                )
                executor.submit(
                    model.fit, feature_data, np.array(target_data)[:, n_step]
                )

        return self

    @validated()
    def predict(self, dataset: Dataset,
        num_samples: Optional[int] = None
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
