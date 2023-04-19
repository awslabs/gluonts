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

from gluonts.dataset.common import Dataset
from gluonts.evaluation import Evaluator
from gluonts.model.predictor import Predictor

from gluonts.nursery.temporal_hierarchical_forecasting.utils.common import (
    TEMPORAL_HIERARCHIES,
)
from gluonts.nursery.temporal_hierarchical_forecasting.utils.utils import (
    get_ts_at_all_levels,
    unpack_forecasts,
)
from gluonts.nursery.temporal_hierarchical_forecasting.utils.gluonts_helper import (
    make_predictions,
    to_dataframe_it,
    truncate_target,
)


def evaluate_predictor(
    predictor: Predictor,
    test_dataset: Dataset,
    freq: str,
    metrics: List[str] = ["mean_wQuantileLoss"],
    evaluate_all_levels: bool = False,
):
    pred_input_it = truncate_target(
        test_dataset,
        prediction_length=predictor.prediction_length,
    )

    forecast_it = make_predictions(
        pred_input=pred_input_it,
        predictor=predictor,
        forecasts_at_all_levels=evaluate_all_levels,
    )
    test_ts_it = to_dataframe_it(test_dataset)

    evaluator = Evaluator(quantiles=(np.arange(20) / 20.0)[1:], num_workers=0)
    if not evaluate_all_levels:
        # Evaluate only the bottom level.
        agg_metrics, _ = evaluator(
            ts_iterator=test_ts_it,
            fcst_iterator=forecast_it,
        )
        metrics_to_return = {
            metric_str: agg_metrics[metric_str] for metric_str in metrics
        }
        return metrics_to_return

    # Evaluate all levels.
    metrics_to_return = evaluate_forecasts_at_all_levels(
        forecast_at_all_levels_it=forecast_it,
        test_ts_at_all_levels_it=get_ts_at_all_levels(
            ts_it=test_ts_it,
            temporal_hierarchy=TEMPORAL_HIERARCHIES[freq],
            prediction_length=predictor.prediction_length,
            target_temporal_hierarchy=TEMPORAL_HIERARCHIES[freq],
        ),
        temporal_hierarchy=TEMPORAL_HIERARCHIES[freq],
        evaluator=evaluator,
        metrics=metrics,
    )
    return metrics_to_return


def evaluate_forecasts_at_all_levels(
    forecast_at_all_levels_it,
    test_ts_at_all_levels_it,
    temporal_hierarchy,
    evaluator,
    metrics: List[str] = ["mean_wQuantileLoss"],
):
    forecast_at_all_levels_unpacked_it = unpack_forecasts(
        forecast_at_all_levels_it=forecast_at_all_levels_it,
        temporal_hierarchy=temporal_hierarchy,
        target_temporal_hierarchy=temporal_hierarchy,
    )

    # First get item metrics for all time series for all frequencies; these are per time series metrics.
    # Then we aggregate the metrics by slicing according to the hierarchy.
    # `metrics_per_ts` is a dataframe where columns contain all item metrics;
    # number of rows = num_levels x num_ts, with the row ordering:
    #  (ts_1_6M, ts_1_2M, ts_1_1M, ts_2_6M, ts_2_2M, ts_2_1M, ...)
    _, metrics_per_ts = evaluator(
        ts_iterator=test_ts_at_all_levels_it,
        fcst_iterator=forecast_at_all_levels_unpacked_it,
    )

    num_levels = len(temporal_hierarchy.agg_multiples)
    metrics_to_return = {}
    for level in range(num_levels):
        agg_metrics_level, _ = evaluator.get_aggregate_metrics(
            metrics_per_ts.iloc[level:None:num_levels]
        )

        for metric_name in metrics:
            metrics_to_return[
                f"Freq_{temporal_hierarchy.freq_strs[level]}_{metric_name}"
            ] = agg_metrics_level[metric_name]

    return metrics_to_return
