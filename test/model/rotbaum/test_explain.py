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
from itertools import chain
from math import isclose
from random import randint
from typing import Tuple, List

import numpy as np
import numpy.random
import pytest
from scipy.ndimage import shift

from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
from gluonts.model.rotbaum import TreeEstimator
from gluonts.model.rotbaum._types import ExplanationResult

logger = logging.getLogger(__name__)


def make_causal_datasets(
    num_ts: int = 5,
    start: str = "2018-01-01",
    freq: str = "D",
    min_length: int = 50,
    max_length: int = 100,
    num_feat_static_cat: int = 3,
    num_past_feat_dynamic_real: int = 1,
) -> Tuple[ListDataset, ListDataset]:
    """
    Generate causal target time series following this formula
    y_t = sum(rts_t-1) + sum(im)
    both rts and im are independent variable generated randomly at each time stamp
    :param num_ts: number of time series to generate
    :param start: start date of all time series
    :param freq: frequency of the time series
    :param min_length: min # of observations per time series
    :param max_length: max # of observations per time series
    :param num_feat_static_cat:
    :param num_past_feat_dynamic_real:
    :return:
    """
    data_iter_train = []
    data_iter_test = []
    assert (
        num_feat_static_cat > 0 and num_past_feat_dynamic_real > 0
    ), "both num_feat_static_cat and num_past_feat_dynamic_real should be provided"
    for k in range(num_ts):
        ts_length = randint(min_length, max_length)
        data_entry_train = {}
        data_entry_train[FieldName.FEAT_STATIC_CAT] = [
            randint(0, 1000) for c in range(num_feat_static_cat)
        ]
        data_entry_train[FieldName.PAST_FEAT_DYNAMIC_REAL] = [
            [randint(0, 1000)] * ts_length
            for k in range(num_past_feat_dynamic_real)
        ]

        data_entry_train[FieldName.START] = start
        # shift the past_feat_dynamic_real by 1 since Y_t depends on X_(t-1)
        shifted_past_feat_dynamic_real = [
            shift(past_feat_dynamic_real, 1, cval=0)
            for past_feat_dynamic_real in data_entry_train[
                FieldName.PAST_FEAT_DYNAMIC_REAL
            ]
        ]
        sum_of_feat_static_cat = sum(
            data_entry_train[FieldName.FEAT_STATIC_CAT]
        )

        data_entry_train[FieldName.TARGET] = (
            np.array(shifted_past_feat_dynamic_real).sum(axis=0)
            + sum_of_feat_static_cat
        ).tolist()

        data_entry_test = data_entry_train.copy()
        data_iter_train.append(data_entry_train)
        data_iter_test.append(data_entry_test)

    return (
        ListDataset(data_iter=data_iter_train, freq=freq),
        ListDataset(data_iter=data_iter_test, freq=freq),
    )


@pytest.mark.parametrize("methods", ["QuantileRegression"])
def test_rotbaum_explain(methods):
    dataset_train, dataset_test = make_causal_datasets(
        num_past_feat_dynamic_real=3, num_feat_static_cat=3
    )
    hps = {
        "freq": "D",
        "prediction_length": 5,
        "quantiles": [0.1, 0.5, 0.9],
        "method": methods,
        "use_past_feat_dynamic_real": True,
    }
    estimator = TreeEstimator.from_inputs(dataset_train, **hps)
    predictor = estimator.train(dataset_train)
    result = predictor.explain().dict()
    logger.info(f"explain result is {result}")
    # check all fields sum up to 1 per prediction horizion
    horizon_sum = []
    for entry in result["quantile_aggregated_result"].values():
        if entry:
            horizon_sum.append(np.sum(entry, axis=0).tolist())
    for timestamp_sum in np.sum(horizon_sum, axis=0):
        assert isclose(timestamp_sum, 1.0)

    # check time quantile aggregated result
    assert isclose(
        np.sum(
            list(
                chain.from_iterable(
                    result["time_quantile_aggregated_result"].values()
                )
            )
        ),
        1.0,
    )
    assert (
        np.sum(
            result["time_quantile_aggregated_result"][
                FieldName.PAST_FEAT_DYNAMIC_REAL
            ]
        )
        >= 0
    )
    assert (
        np.sum(
            result["time_quantile_aggregated_result"][
                FieldName.FEAT_STATIC_CAT
            ]
        )
        >= 0
    )
    assert (
        np.sum(result["time_quantile_aggregated_result"][FieldName.TARGET])
        >= 0
    )
