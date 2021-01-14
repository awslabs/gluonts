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

from typing import Iterator

import torch

from gluonts.model.forecast_generator import (
    recursively_zip_arrays,
    make_distribution_forecast,
    DistributionForecastGenerator,  # this is for analogy with gluonts.mx.model.forecast_generator
)
from gluonts.torch.model.forecast import DistributionForecast


@recursively_zip_arrays.register(torch.Tensor)
def _(x: torch.Tensor) -> Iterator[torch.Tensor]:
    for i in range(x.shape[0]):
        yield x[i]


@make_distribution_forecast.register(torch.distributions.Distribution)
def _(
    distr: torch.distributions.Distribution, *args, **kwargs
) -> DistributionForecast:
    return DistributionForecast(distr, *args, **kwargs)
