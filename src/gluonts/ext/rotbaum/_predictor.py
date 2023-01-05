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
from toolz import first

import numpy as np
import pandas as pd
from itertools import compress

from gluonts.core.component import validated
from gluonts.dataset.common import Dataset
from gluonts.dataset.util import forecast_start
from gluonts.model.forecast import Forecast
from gluonts.model.forecast_generator import log_once
from gluonts.model.predictor import RepresentablePredictor

from ._model import QRF, QRX, QuantileReg
from ._preprocess import Cardinality, PreprocessOnlyLagFeatures
from ._types import FeatureImportanceResult, ExplanationResult

logger = logging.getLogger(__name__)


class RotbaumForecast(Forecast):
    """
    Implements the quantile function in Forecast for TreePredictor, as well as
    a new estimate_dists function for estimating a sampling of the conditional
    distribution of the value of each of the steps in the forecast horizon
    (independently).
    """

    @validated()
    def __init__(
        self,
        models: List,
        featurized_data: List,
        start_date: pd.Period,
        prediction_length: int,
    ):
        self.models = models
        self.featurized_data = featurized_data
        self.start_date = start_date
        self.prediction_length = prediction_length
        self.item_id = None
        self.lead_time = None

    def quantile(self, q: float) -> np.ndarray:
        """
        Returns np.array, where the i^th entry is the estimate of the q
        quantile of the conditional distribution of the value of the i^th step
        in the forecast horizon.
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

    def estimate_dists(self) -> np.ndarray:
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
    horizon.

    (In other words, there's a total of prediction_length many models being
    trained. In particular, this predictor does not learn a multivariate
    distribution.) The list of these models is saved under self.model_list.
    """

    @validated()
    def __init__(
        self,
        freq: str,
        prediction_length: int,
        n_ignore_last: int = 0,
        lead_time: int = 0,
        max_n_datapts: int = 1000000,
        min_bin_size: int = 100,  # Used only for "QRX" method.
        context_length: Optional[int] = None,
        use_feat_static_real: bool = False,
        use_past_feat_dynamic_real: bool = False,
        use_feat_dynamic_real: bool = False,
        use_feat_dynamic_cat: bool = False,
        cardinality: Cardinality = "auto",
        one_hot_encode: bool = False,
        model_params: Optional[dict] = None,
        max_workers: Optional[int] = None,
        method: str = "QRX",
        quantiles=None,  # Used only for "QuantileRegression" method.
        subtract_mean: bool = True,
        count_nans: bool = False,
        model=None,
        seed=None,
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
            use_past_feat_dynamic_real=use_past_feat_dynamic_real,
            use_feat_dynamic_real=use_feat_dynamic_real,
            use_feat_dynamic_cat=use_feat_dynamic_cat,
            cardinality=cardinality,
            one_hot_encode=one_hot_encode,
            subtract_mean=subtract_mean,
            count_nans=count_nans,
            seed=seed,
        )

        assert (
            context_length is None or context_length > 0
        ), "The value of `context_length` should be > 0"

        # TODO: Figure out how to include 'auto' with no feat_static_cat in
        # this check
        assert (
            prediction_length > 0
            or use_feat_dynamic_cat
            or use_past_feat_dynamic_real
            or use_feat_dynamic_real
            or use_feat_static_real
            or cardinality != "ignore"
        ), (
            "The value of `prediction_length` should be > 0 or there should be"
            " features for model training and prediction "
        )

        self.model_params = model_params if model_params else {}
        self.prediction_length = prediction_length
        self.freq = freq
        self.max_workers = max_workers
        self.min_bin_size = min_bin_size
        self.quantiles = quantiles
        self.model = model
        self.model_list = None

        logger.info(
            "If using the Evaluator class with a TreePredictor, set"
            " num_workers=0."
        )

    def train(
        self,
        training_data,
        train_QRX_only_using_timestep: int = -1,  # If not -1 and self.method
        # == 'QRX', this will use only the train_QRX_only_using_timestep^th
        # timestep in the forecast horizon to create the partition.
    ):
        assert training_data
        if self.preprocess_object.use_feat_dynamic_real:
            assert (
                len(first(training_data)["feat_dynamic_real"][0])
                == len(first(training_data)["target"])
                + self.preprocess_object.forecast_horizon
            )
        if self.preprocess_object.use_past_feat_dynamic_real:
            assert len(
                first(training_data)["past_feat_dynamic_real"][0]
            ) == len(first(training_data)["target"])
        assert self.freq is not None
        if first(training_data)["start"].freq is not None:
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
                    min_bin_size=self.min_bin_size,
                    model=self.model,
                )
                for _ in range(n_models)
            ]
        if train_QRX_only_using_timestep != -1:
            assert (
                0
                <= train_QRX_only_using_timestep
                <= self.preprocess_object.forecast_horizon - 1
            )
            logger.info(
                "Training model for step no."
                f" {train_QRX_only_using_timestep} in the forecast horizon"
            )
            self.model_list[train_QRX_only_using_timestep].fit(
                feature_data,
                np.array(target_data)[:, train_QRX_only_using_timestep],
            )
            self.model_list = [
                QRX(
                    xgboost_params=self.model_params,
                    min_bin_size=self.min_bin_size,
                    model=self.model_list[train_QRX_only_using_timestep].model,
                )
                if i != train_QRX_only_using_timestep
                else self.model_list[i]
                for i in range(n_models)
            ]
            target_data = np.array(target_data)
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_workers
            ) as executor:
                for n_step, model in enumerate(self.model_list):
                    current_target_data = target_data[:, n_step]
                    indices = ~np.isnan(current_target_data)
                    assert sum(indices), (
                        f"in timestamp {n_step} all values "
                        f"in train were nan"
                    )
                    if n_step != train_QRX_only_using_timestep:
                        logger.info(
                            f"Training model for step no. {n_step + 1} in the "
                            "forecast"
                            " horizon"
                        )
                        executor.submit(
                            model.fit,
                            list(compress(feature_data, indices)),
                            current_target_data[indices],
                            model_is_already_trained=True,
                        )
        else:
            target_data = np.array(target_data)
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_workers
            ) as executor:
                for n_step, model in enumerate(self.model_list):
                    current_target_data = target_data[:, n_step]
                    indices = ~np.isnan(current_target_data)
                    assert sum(indices), (
                        f"in timestamp {n_step} all values "
                        f"in train were nan"
                    )
                    logger.info(
                        f"Training model for step no. {n_step + 1} in the"
                        " forecast horizon"
                    )
                    executor.submit(
                        model.fit,
                        list(compress(feature_data, indices)),
                        current_target_data[indices],
                    )
        return self

    def predict(
        self, dataset: Dataset, num_samples: Optional[int] = None
    ) -> Iterator[Forecast]:
        """
        Returns a dictionary taking each quantile to a list of floats, which
        are the predictions for that quantile as you run over (time_steps,
        time_series) lexicographically.

        So: first it would give
        the quantile prediction for the first time step for all time series,
        then the second time step for all time series ˜˜ , and so forth.
        """
        context_length = self.preprocess_object.context_window_size

        if num_samples:
            log_once(
                "Forecast is not sample based. Ignoring parameter"
                " `num_samples` from predict method."
            )
        if not self.model_list:
            logger.error("model_list is empty during prediction")

        for ts in dataset:
            featurized_data = self.preprocess_object.make_features(
                ts, starting_index=len(ts["target"]) - context_length
            )
            yield RotbaumForecast(
                self.model_list,
                [featurized_data],
                start_date=forecast_start(ts),
                prediction_length=self.prediction_length,
            )

    def explain(
        self, importance_type: str = "gain", percentage: bool = True
    ) -> ExplanationResult:
        """
        This function only works for self.method == "QuantileRegression",
        and uses lightgbm's feature importance functionality. It takes the
        mean feature importance across quantiles and timestamps in the
        forecast horizon; and then adds these mean values across all of the
        feature coordinates that are associated to "target",
        "feat_static_real", "feat_static_cat", "past_feat_dynamic_real",
        "feat_dynamic_real", "feat_dynamic_cat"

        Parameters
        ----------
        importance_type: str
            Either "gain" or "split". Since for the models that predict
            timestamps that are further along in the forecast horizon are
            expected to perform less well, it is desirable to give less
            weight to those models compared to the ones that make
            predictions for timestamps closer to the forecast horizon.
            "split" will give equal weight, and is therefore less desirable;
            whereas "gain" will naturally give less weight to models that
            perform less well.

        percentage: bool
            If results should be in percentage format and sum up to 1. Default is True

        Returns
        -------
        ExplanationResult
        """
        assert self.method == "QuantileRegression", (
            "the explain method is "
            "only currently "
            "supported for "
            "QuantileRegression"
        )
        importances = np.array(
            [
                [
                    self.model_list[time_stamp]
                    .models[quantile]
                    .booster_.feature_importance(
                        importance_type=importance_type
                    )
                    for time_stamp in range(self.prediction_length)
                ]
                for quantile in self.quantiles
            ]
        ).transpose((2, 1, 0))
        # The shape is: (features, pred_length, quantiles)
        importances = importances.mean(axis=2)  # Average over quantiles
        # The shape of importances is: (features, pred_length)

        if percentage:
            importances /= importances.sum(axis=0)
        dynamic_length = self.preprocess_object.dynamic_length
        num_feat_static_real = self.preprocess_object.num_feat_static_real
        num_feat_static_cat = self.preprocess_object.num_feat_static_cat
        num_past_feat_dynamic_real = (
            self.preprocess_object.num_past_feat_dynamic_real
        )
        num_feat_dynamic_real = self.preprocess_object.num_feat_dynamic_real
        num_feat_dynamic_cat = self.preprocess_object.num_feat_dynamic_cat
        coordinate_map = {}
        coordinate_map["target"] = (0, dynamic_length)
        coordinate_map["feat_static_real"] = [
            (dynamic_length + i, dynamic_length + i + 1)
            for i in range(num_feat_static_real)
        ]
        coordinate_map["feat_static_cat"] = []
        static_cat_features_so_far = 0
        cardinality = (
            self.preprocess_object.cardinality
            if (
                self.preprocess_object.cardinality
                and self.preprocess_object.one_hot_encode
            )
            else [1] * num_feat_static_cat
        )

        for i in range(num_feat_static_cat):
            coordinate_map["feat_static_cat"].append(
                (
                    dynamic_length
                    + num_feat_static_real
                    + static_cat_features_so_far,
                    dynamic_length
                    + num_feat_static_real
                    + static_cat_features_so_far
                    + cardinality[i],
                )
            )
            static_cat_features_so_far += cardinality[i]

        coordinate_map["past_feat_dynamic_real"] = [
            (
                num_feat_static_real
                + static_cat_features_so_far
                + (i + 1) * dynamic_length,
                num_feat_static_real
                + static_cat_features_so_far
                + (i + 2) * dynamic_length,
            )
            for i in range(num_past_feat_dynamic_real)
        ]
        coordinate_map["feat_dynamic_real"] = [
            (
                num_feat_static_real
                + static_cat_features_so_far
                + (num_past_feat_dynamic_real + 1) * dynamic_length
                + i * (dynamic_length + self.prediction_length),
                num_feat_static_real
                + static_cat_features_so_far
                + (num_past_feat_dynamic_real + 1) * dynamic_length
                + (i + 1) * (dynamic_length + self.prediction_length),
            )
            for i in range(num_feat_dynamic_real)
        ]
        coordinate_map["feat_dynamic_cat"] = [
            (
                num_feat_static_real
                + static_cat_features_so_far
                + (num_past_feat_dynamic_real + 1) * dynamic_length
                + num_feat_dynamic_real
                * (dynamic_length + self.prediction_length)
                + i,
                num_feat_static_real
                + static_cat_features_so_far
                + (num_past_feat_dynamic_real + 1) * dynamic_length
                + num_feat_dynamic_real
                * (dynamic_length + self.prediction_length)
                + i
                + 1,
            )
            for i in range(num_feat_dynamic_cat)
        ]
        logger.info(
            f"coordinate_map from the preprocessor is: {coordinate_map}"
        )
        logger.info(f"shape of importance matrix is: {importances.shape}")
        assert (
            sum(
                [
                    sum([coor[1] - coor[0] for coor in coordinate_map[key]])
                    for key in coordinate_map
                    if key != "target"
                ]
            )
            + coordinate_map["target"][1]
            - coordinate_map["target"][0]
        ) == importances.shape[
            0
        ]  # Testing that we covered all of coordinates

        quantile_aggregated_importance_result = FeatureImportanceResult(
            target=np.expand_dims(
                importances[
                    coordinate_map["target"][0] : coordinate_map["target"][1],
                    :,
                ].sum(axis=0),
                axis=0,
            ).tolist(),
            feat_static_real=[
                importances[start:end, :].sum(axis=0).tolist()
                for start, end in coordinate_map["feat_static_real"]
            ],
            feat_static_cat=[
                importances[start:end, :].sum(axis=0).tolist()
                for start, end in coordinate_map["feat_static_cat"]
            ],
            past_feat_dynamic_real=[
                importances[start:end, :].sum(axis=0).tolist()
                for start, end in coordinate_map["past_feat_dynamic_real"]
            ],
            feat_dynamic_real=[
                importances[start:end, :].sum(axis=0).tolist()
                for start, end in coordinate_map["feat_dynamic_real"]
            ],
            feat_dynamic_cat=[
                importances[start:end, :].sum(axis=0).tolist()
                for start, end in coordinate_map["feat_dynamic_cat"]
            ],
        )

        explain_result = ExplanationResult(
            time_quantile_aggregated_result=quantile_aggregated_importance_result.mean(
                axis=1
            ),
            quantile_aggregated_result=quantile_aggregated_importance_result,
        )
        logger.info(
            f"explain result with importance_type: {importance_type} and percentage: {percentage} is {explain_result.dict()}"
        )
        return explain_result
