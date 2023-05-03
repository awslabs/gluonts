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

from copy import deepcopy
from typing import (
    ChainMap,
    Iterator,
    Optional,
    Union,
)
from dataclasses import dataclass

import numpy as np

from gluonts.dataset.repository import get_dataset
from gluonts.dataset.split import TestData, split
from gluonts.ext.naive_2 import naive_2
from gluonts.time_feature.seasonality import get_seasonality
from gluonts.model.forecast import Quantile, SampleForecast
from gluonts.model.seasonal_naive import SeasonalNaivePredictor
from gluonts.evaluation import Evaluator
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.ev import (
    seasonal_error,
    MAPE,
    MASE,
    MSE,
    MSIS,
    SMAPE,
    Coverage,
    SumAbsoluteError,
    SumQuantileLoss,
    mean_absolute_label,
    sum_absolute_label,
    ND,
    NRMSE,
    RMSE,
    MAECoverage,
    MeanSumQuantileLoss,
    MeanWeightedSumQuantileLoss,
    WeightedSumQuantileLoss,
    OWA,
)


@dataclass
class SampleForecastBatch:
    samples: np.ndarray
    start_date: list
    item_id: Optional[list] = None
    info: Optional[list] = None

    def __post_init__(self):
        self._sorted_samples_value = None
        if self.item_id is None:
            self.item_id = [None for _ in self.start_date]
        if self.info is None:
            self.info = [None for _ in self.start_date]

    @property
    def _sorted_samples(self) -> np.ndarray:
        if self._sorted_samples_value is None:
            self._sorted_samples_value = np.sort(self.samples, axis=1)
        return self._sorted_samples_value

    def __iter__(self) -> Iterator[SampleForecast]:
        for sample, start_date, item_id, info in zip(
            self.samples, self.start_date, self.item_id, self.info
        ):
            yield SampleForecast(
                sample,
                start_date=start_date,
                item_id=item_id,
                info=info,
            )

    @property
    def num_samples(self) -> int:
        return self.samples.shape[1]

    @property
    def mean(self) -> np.ndarray:
        return np.mean(self.samples, axis=1)

    def quantile(self, q: Union[float, str]) -> np.ndarray:
        q = Quantile.parse(q).value
        sample_idx = int(np.round((self.num_samples - 1) * q))
        return self._sorted_samples[:, sample_idx, :]

    def __getitem__(self, name):
        if name == "mean":
            return self.mean
        elif name == "median":
            return self.median

        return self.quantile(name)


def get_data_batches(predictor, test_data):
    forecasts = predictor.predict(dataset=test_data.input)
    for input_, label, forecast in zip(
        test_data.input, test_data.label, forecasts
    ):
        forecast_batch = SampleForecastBatch(
            samples=np.array([forecast.samples]),
            start_date=[forecast.start_date],
            item_id=[forecast.item_id],
            info=[forecast.info],
        )

        seasonality = get_seasonality(freq=forecast.start_date.freqstr)
        freq = forecast.start_date.freqstr
        other_data = {
            "label": np.array([label["target"]]),
            "seasonal_error": np.array(
                [seasonal_error(input_["target"], seasonality=seasonality)]
            ),
            "naive_2": np.array(
                [naive_2(input_["target"], len(label["target"]), freq=freq)]
            ),
        }

        yield ChainMap(other_data, forecast_batch)


def evaluate(metrics, data_batches, axis):
    evaluators = {}
    for metric in metrics:
        evaluator = metric(axis=axis)
        evaluators[evaluator.name] = evaluator

    for data_batch in iter(data_batches):
        for evaluator in evaluators.values():
            evaluator.update(data_batch)

    return {
        metric_name: evaluator.get()
        for metric_name, evaluator in evaluators.items()
    }


def get_new_metrics(test_data, predictor, quantile_levels):
    quantiles = [Quantile.parse(q) for q in quantile_levels]

    item_metrics_to_evaluate = [
        sum_absolute_label,
        SumAbsoluteError(),
        *(SumQuantileLoss(q=quantile.value) for quantile in quantiles),
        mean_absolute_label,
        MSE(),
        MASE(),
        MAPE(),
        SMAPE(),
        MSIS(),
        *(Coverage(q=quantile.value) for quantile in quantiles),
    ]
    aggregated_metrics_to_evaluate = [
        RMSE(),
        NRMSE(),
        ND(),
        *(WeightedSumQuantileLoss(q=quantile.value) for quantile in quantiles),
        MeanSumQuantileLoss([quantile.value for quantile in quantiles]),
        MeanWeightedSumQuantileLoss(
            [quantile.value for quantile in quantiles]
        ),
        MAECoverage([quantile.value for quantile in quantiles]),
        OWA(),
    ]

    # mask invalid values
    masked_dataset = []
    for data_entry in test_data.dataset:
        masked_entry = deepcopy(data_entry)
        masked_entry["target"] = np.ma.masked_invalid(data_entry["target"])
        masked_dataset.append(masked_entry)

    masked_test_data = TestData(
        dataset=masked_dataset,
        splitter=test_data.splitter,
        prediction_length=test_data.prediction_length,
        windows=test_data.windows,
        distance=test_data.distance,
        max_history=test_data.max_history,
    )

    # evaluate
    data_batches = list(get_data_batches(predictor, masked_test_data))
    item_metrics = evaluate(item_metrics_to_evaluate, data_batches, axis=1)
    aggregated_metrics = evaluate(
        aggregated_metrics_to_evaluate, data_batches, axis=None
    )

    # rename metrics to match naming of old evaluation approach
    all_metrics = {
        "seasonal_error": np.ma.mean(
            np.stack([data["seasonal_error"] for data in data_batches])
        ),
        "abs_target_mean": np.ma.mean(item_metrics["mean_absolute_label"]),
        **{
            f"Coverage[{quantile}]": np.ma.mean(
                item_metrics[f"coverage[{quantile.value}]"]
            )
            for quantile in quantiles
        },
        "abs_error": np.ma.sum(item_metrics["sum_absolute_error"]),
        "abs_target_sum": np.ma.sum(item_metrics["sum_absolute_label"]),
        **{
            f"QuantileLoss[{quantile}]": np.ma.sum(
                item_metrics[f"sum_quantile_loss[{quantile.value}]"]
            )
            for quantile in quantiles
        },
        **{
            f"wQuantileLoss[{quantile}]": np.ma.sum(
                aggregated_metrics[
                    f"weighted_sum_quantile_loss[{quantile.value}]"
                ]
            )
            for quantile in quantiles
        },
        "mean_absolute_QuantileLoss": aggregated_metrics[
            "mean_sum_quantile_loss"
        ],
        "mean_wQuantileLoss": aggregated_metrics[
            "mean_weighted_sum_quantile_loss"
        ],
        "MAE_Coverage": aggregated_metrics["MAE_coverage"],
    }

    for metric_name in ["MSE", "MASE", "MAPE", "sMAPE", "MSIS"]:
        all_metrics[metric_name] = np.ma.mean(item_metrics[metric_name])

    for metric_name in ["RMSE", "NRMSE", "ND", "OWA"]:
        all_metrics[metric_name] = aggregated_metrics[metric_name]

    return all_metrics


def get_old_metrics(dataset, predictor, quantile_levels):
    forecast_it, ts_it = make_evaluation_predictions(dataset, predictor)
    evaluator = Evaluator(quantiles=quantile_levels, calculate_owa=True)
    aggregated_metrics, _ = evaluator(ts_it, forecast_it)

    return aggregated_metrics


def test_against_former_evaluator():
    dataset = get_dataset("exchange_rate")

    prediction_length = dataset.metadata.prediction_length
    freq = dataset.metadata.freq

    _, test_template = split(dataset=dataset.test, offset=-prediction_length)
    test_data = test_template.generate_instances(
        prediction_length=prediction_length
    )

    predictor = SeasonalNaivePredictor(
        prediction_length=prediction_length, freq=freq
    )

    quantile_levels = (0.1, 0.5, 0.9)
    ev_result = get_new_metrics(test_data, predictor, quantile_levels)
    evaluation_result = get_old_metrics(
        dataset.test, predictor, quantile_levels
    )

    for metric_name in ev_result.keys():
        np.testing.assert_allclose(
            ev_result[metric_name],
            evaluation_result[metric_name],
            err_msg=f"Metric '{metric_name}' is {ev_result[metric_name]} but "
            f"should be {evaluation_result[metric_name]}",
        )
