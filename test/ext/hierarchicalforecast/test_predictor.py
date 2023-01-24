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

from pathlib import Path
from tempfile import TemporaryDirectory
import itertools
import pytest

from statsforecast.models import AutoARIMA, AutoETS
from hierarchicalforecast.methods import (
    BottomUp,
    ERM,
    MiddleOut,
    MinTrace,
    OptimalCombination,
    TopDown,
)

from gluonts.evaluation import MultivariateEvaluator, backtest_metrics
from gluonts.model import Predictor
from gluonts.core import serde
from gluonts.ext.hierarchicalforecast import (
    ModelConfig,
    HierarchicalForecastPredictor,
)

TOLERANCE = 10.0
SEASON_LENGTH = 2
QUANTILE_LEVELS = [0.5, 0.1, 0.9, 0.95]
PREDICTION_LENGTH = 1
TS_NAMES = ["total", "A", "B", "A-1", "A-2", "B-1", "B-2"]
TAGS = {
    "L0": ["total"],
    "L0/L1": ["A", "B"],
    "L0/L1/L2": ["A-1", "A-2", "B-1", "B-2"],
}


def generate_test_inputs(season_length: int) -> list():
    base_models = [AutoARIMA, AutoETS]
    model_params = {"season_length": season_length}
    reconcilers = {
        "BottomUp": BottomUp,
        "TopDown": TopDown,
        "MiddleOut": MiddleOut,
        "MinTrace": MinTrace,
        "OptimalCombination": OptimalCombination,
        "ERM": ERM,
    }

    top_down_proportions = [
        "forecast_proportions",
        "average_proportions",
        "proportion_averages",
    ]
    mintrace_methods = ["ols", "wls_struct", "wls_var", "mint_shrink"]
    optimalcombination_methods = ["ols", "wls_struct"]
    erm_methods = ["closed", "reg", "reg_bu"]

    reconciler_params = {
        "BottomUp": [{}],
        "TopDown": [{"method": method} for method in top_down_proportions],
        "MiddleOut": [
            {"top_down_method": method, "middle_level": "L0/L1"}
            for method in top_down_proportions
        ],
        "MinTrace": [{"method": method} for method in mintrace_methods],
        "OptimalCombination": [
            {"method": method} for method in optimalcombination_methods
        ],
        "ERM": [{"method": method} for method in erm_methods],
    }

    test_input = []
    for base_model in base_models:
        for reconciler in reconcilers:
            test_input.extend(
                list(
                    itertools.product(
                        [base_model],
                        [reconcilers[reconciler]],
                        [model_params],
                        reconciler_params[reconciler],
                    )
                )
            )

    return test_input


test_input = generate_test_inputs(SEASON_LENGTH)


@pytest.mark.parametrize(
    "base_model, reconciler, model_params, reconciler_params",
    test_input,
)
def test_predictor_serialization(
    sine7, base_model, reconciler, model_params, reconciler_params
):
    train_datasets = sine7(nonnegative=True)

    (train_dataset, test_dataset, metadata) = (
        train_datasets.train,
        train_datasets.test,
        train_datasets.metadata,
    )

    S = metadata.S

    predictor = HierarchicalForecastPredictor(
        prediction_length=PREDICTION_LENGTH,
        base_model=base_model,
        reconciler=reconciler,
        S=S,
        tags=TAGS,
        ts_names=TS_NAMES,
        quantile_levels=QUANTILE_LEVELS,
        model_params=model_params,
        reconciler_params=reconciler_params,
    )
    assert predictor == serde.decode(serde.encode(predictor))

    with TemporaryDirectory() as temp_dir:
        predictor.serialize(Path(temp_dir))
        predictor_copy = Predictor.deserialize(Path(temp_dir))

    assert predictor_copy == predictor


@pytest.mark.parametrize(
    "base_model, reconciler, model_params, reconciler_params",
    test_input,
)
def test_predictor_working(
    sine7, base_model, reconciler, model_params, reconciler_params
):
    train_datasets = sine7(
        prediction_length=PREDICTION_LENGTH,
        nonnegative=True,
        seq_length=25,
        bias=1,
    )

    (train_dataset, test_dataset, metadata) = (
        train_datasets.train,
        train_datasets.test,
        train_datasets.metadata,
    )

    S = metadata.S

    predictor = HierarchicalForecastPredictor(
        prediction_length=PREDICTION_LENGTH,
        base_model=base_model,
        reconciler=reconciler,
        S=S,
        tags=TAGS,
        ts_names=TS_NAMES,
        quantile_levels=QUANTILE_LEVELS,
        model_params=model_params,
        reconciler_params=reconciler_params,
    )

    agg_metrics, item_metrics = backtest_metrics(
        test_dataset=test_dataset,
        predictor=predictor,
        evaluator=MultivariateEvaluator(
            quantiles=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
        ),
    )

    assert agg_metrics["mean_wQuantileLoss"] < TOLERANCE
    assert agg_metrics["NRMSE"] < TOLERANCE
    assert agg_metrics["RMSE"] < TOLERANCE


@pytest.mark.parametrize(
    "quantile_levels, intervals, forecast_keys, statsforecast_keys",
    [
        (
            [0.5, 0.2, 0.6, 0.9, 0.8, 0.1],
            [0, 20, 60, 80],
            ["mean", "0.5", "0.2", "0.6", "0.9", "0.8", "0.1"],
            ["mean", "lo-0", "lo-60", "hi-20", "hi-80", "hi-60", "lo-80"],
        ),
    ],
)
def test_model_config(
    quantile_levels, intervals, forecast_keys, statsforecast_keys
):
    config = ModelConfig(quantile_levels=quantile_levels)
    assert config.intervals == intervals
    assert config.forecast_keys == forecast_keys
    assert config.statsforecast_keys == statsforecast_keys
