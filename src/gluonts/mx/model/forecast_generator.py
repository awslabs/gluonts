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

from typing import Any, Callable, Iterator, List, Optional

import mxnet as mx

from gluonts.model.forecast_generator import (
    recursively_zip_arrays,
    make_distribution_forecast,
    DistributionForecastGenerator,  # this is for backward compatibility
)
from gluonts.mx.distribution import Distribution
from gluonts.mx.model.forecast import DistributionForecast


@recursively_zip_arrays.register(mx.nd.NDArray)
def _(x: mx.nd.NDArray) -> Iterator[mx.nd.NDArray]:
    for i in range(x.shape[0]):
        yield x[i]


@make_distribution_forecast.register(Distribution)
def _(distr: Distribution, *args, **kwargs) -> DistributionForecast:
    return DistributionForecast(distr, *args, **kwargs)
