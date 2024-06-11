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

from typing import List, Optional, Tuple

import torch

from gluonts.core.component import validated
from gluonts.model.forecast_generator import (
    ForecastGenerator,
    QuantileForecastGenerator,
)

from .distribution_output import Output


class QuantileOutput(Output):
    @validated()
    def __init__(self, quantiles: List[float]) -> None:
        assert len(quantiles) > 0
        assert all(0.0 < q < 1.0 for q in quantiles)
        self._quantiles = sorted(quantiles)
        self.num_quantiles = len(self._quantiles)
        self.args_dim = {"outputs": self.num_quantiles}

    @property
    def forecast_generator(self) -> ForecastGenerator:
        return QuantileForecastGenerator(quantiles=self.quantiles)

    @property
    def event_shape(self) -> Tuple:
        return ()

    def domain_map(self, *args: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        return args

    @property
    def quantiles(self) -> List[float]:
        return self._quantiles

    def loss(
        self,
        target: torch.Tensor,
        distr_args: Tuple[torch.Tensor, ...],
        loc: Optional[torch.Tensor] = None,
        scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        (quantile_preds,) = distr_args
        if scale is not None:
            quantile_preds = quantile_preds * scale.unsqueeze(-1)
        if loc is not None:
            quantile_preds = quantile_preds + loc.unsqueeze(-1)

        target = target.unsqueeze(-1)
        quantiles = torch.as_tensor(
            self.quantiles,
            device=quantile_preds.device,
            dtype=quantile_preds.dtype,
        )
        return 2 * (
            (target - quantile_preds)
            * ((target <= quantile_preds).float() - quantiles)
        ).abs().mean(dim=-1)
