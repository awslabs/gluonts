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

import numpy as np

from gluonts.ev_with_decorators.api import Input, metric


# POINT METRICS (two-dimensional in univariate case)


@metric(Input("target"))
def abs_target(target):
    return np.abs(target)


@metric(Input("prediction_mean"))
def abs_prediction_mean(prediction_mean):
    return np.abs(prediction_mean)


@metric(Input("target"), Input("prediction_mean"))
def error_wrt_mean(target, prediction_mean):
    return target - prediction_mean


@metric(Input("target"), Input("prediction_mean"))
def error_wrt_median(target, prediction_mean):
    return target - prediction_mean


@metric(error_wrt_median)
def abs_error(error_wrt_median):
    return np.abs(error_wrt_median)


@metric(error_wrt_mean)
def squared_error(error_wrt_mean):
    return error_wrt_mean * error_wrt_mean


# LOCAL METRICS (one-dimensional in univariate case)


@metric(squared_error)
def mse(squared_error):
    return np.mean(squared_error, axis=1)


@metric(mse)
def rmse(mse):
    return np.sqrt(mse)


@metric(abs_error)
def abs_error_sum(abs_error):
    return np.sum(abs_error, axis=1)


@metric(abs_error)
def abs_target_sum(abs_target):
    return np.sum(abs_target, aixs=1)


@metric(abs_error)
def abs_target_mean(abs_target):
    return np.mean(abs_target, aixs=1)


@metric(abs_error, abs_target)
def mape(abs_error, abs_target):
    return np.mean(abs_error / abs_target, axis=1)


@metric(abs_error, abs_target)
def smape(abs_error, abs_target):
    return 2 * np.mean(abs_error / (abs_target + abs_prediction_mean), axis=1)


@metric(abs_error, abs_target_sum)
def nd(abs_error, abs_target_sum):
    return np.sum(abs_error / abs_target_sum, axis=1)


# GLOBAL METRICS (a single number in univariate case)


@metric(mse)
def mean_mse(mse):
    return np.mean(mse)


# GENERATIVE METRICS (metrics that use extra parameters)


def coverage(q: float):
    @metric(
        Input("target"), Input("quantile_predictions"), name=f"coverage[{q}]"
    )
    def coverage_for_q(target, quantile_predictions):
        return np.mean(target < quantile_predictions[q], axis=1)

    return coverage_for_q


def quantile_loss(q: float):
    @metric(
        Input("target"),
        Input("quantile_predictions"),
        name=f"quantile_loss[{q}]",
    )
    def quantile_loss_for_q(target, quantile_predictions):
        return 2 * np.abs(
            (target - quantile_predictions[q])
            * ((quantile_predictions[q] >= target) - q)
        )

    return quantile_loss_for_q
