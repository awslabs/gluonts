from typing import Dict, List, Optional

from gluonts.model.forecast_generator import (
    ForecastGenerator,
    QuantileForecastGenerator,
)

import torch
from torch.distributions import Distribution, constraints

from .distribution_output import Output


class QuantileDistribution(Distribution):
    arg_constraints: Dict[str, constraints.Constraint] = {}

    def __init__(
        self,
        outputs: torch.Tensor,
        quantiles: torch.Tensor,
        validate_args=None,
    ):
        self.outputs = outputs
        self.quantiles = torch.tensor(
            quantiles, device=outputs.device, dtype=outputs.dtype
        )
        super().__init__(
            batch_shape=outputs.shape[:-1], validate_args=validate_args
        )

    def quantile_loss(self, value: torch.Tensor) -> torch.Tensor:
        value = value.unsqueeze(-1)
        return 2 * (
            (value - self.outputs)
            * ((value <= self.outputs).float() - self.quantiles)
        ).abs().sum(dim=-1)


class QuantileOutput(Output):
    def __init__(self, quantiles: List[float]) -> None:
        assert len(quantiles) > 0
        assert all(0.0 < q < 1.0 for q in quantiles)
        self._quantiles = quantiles
        self.num_quantiles = len(self._quantiles)
        self.args_dim = {"outputs": self.num_quantiles}

    @property
    def forecast_generator(self) -> ForecastGenerator:
        return QuantileForecastGenerator(quantiles=self.quantiles)

    @property
    def quantiles(self) -> List[float]:
        return self._quantiles

    def domain_map(self, *args: torch.Tensor):
        return args

    def distribution(
        self,
        distr_args,
        loc: Optional[torch.Tensor] = None,
        scale: Optional[torch.Tensor] = None,
    ) -> Distribution:
        (outputs,) = distr_args
        if scale is not None:
            outputs = outputs * scale.unsqueeze(-1)
        if loc is not None:
            outputs = outputs + loc.unsqueeze(-1)
        return QuantileDistribution(outputs=outputs, quantiles=self.quantiles)
