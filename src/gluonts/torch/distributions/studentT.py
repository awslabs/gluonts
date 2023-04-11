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

from typing import Dict, Tuple, Union

import torch
import torch.nn.functional as F
from scipy.stats import t as ScipyStudentT
from torch.distributions import StudentT as TorchStudentT

from .distribution_output import DistributionOutput
from gluonts.util import lazy_property


class StudentT(TorchStudentT):
    """Student's t-distribution parametrized by degree of freedom `df`,
    mean `loc` and scale `scale`.

    Based on torch.distributions.StudentT, with added `cdf` and `icdf` methods.
    """

    def __init__(
        self,
        df: Union[float, torch.Tensor],
        loc: Union[float, torch.Tensor] = 0.0,
        scale: Union[float, torch.Tensor] = 1.0,
        validate_args=None,
    ):
        super().__init__(
            df=df, loc=loc, scale=scale, validate_args=validate_args
        )

    def cdf(self, value: torch.Tensor) -> torch.Tensor:
        if self._validate_args:
            self._validate_sample(value)
        result = self.scipy_student_t.cdf(value.detach().cpu().numpy())
        return torch.tensor(result, device=value.device, dtype=value.dtype)

    def icdf(self, value: torch.Tensor) -> torch.Tensor:
        result = self.scipy_student_t.ppf(value.detach().cpu().numpy())
        return torch.tensor(result, device=value.device, dtype=value.dtype)

    @lazy_property
    def scipy_student_t(self):
        return ScipyStudentT(
            df=self.df.detach().cpu().numpy(),
            loc=self.loc.detach().cpu().numpy(),
            scale=self.scale.detach().cpu().numpy(),
        )


class StudentTOutput(DistributionOutput):
    args_dim: Dict[str, int] = {"df": 1, "loc": 1, "scale": 1}
    distr_cls: type = StudentT

    @classmethod
    def domain_map(
        cls, df: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor
    ):
        epsilon = torch.finfo(scale.dtype).eps
        scale = F.softplus(scale).clamp_min(epsilon)
        df = 2.0 + F.softplus(df)
        return df.squeeze(-1), loc.squeeze(-1), scale.squeeze(-1)

    @property
    def event_shape(self) -> Tuple:
        return ()
