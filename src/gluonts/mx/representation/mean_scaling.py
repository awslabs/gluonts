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

from gluonts.core.component import validated
from gluonts.mx import Tensor

from .representation import Representation


class MeanScaling(Representation):
    """
    A class representing a mean scaling approach.
    Inputs are simply rescaled based on their mean.

    Parameters
    ----------
    minimum_scale
        The minimum value to which re-scaled values will be clipped to.
        (default: 1e-10)
    clip_max
        The maximum value to which re-scaled values will be clipped to.
        Negative values will be clipped at -clip_max and positive values at clip_max.
        (default: None)
    """

    @validated()
    def __init__(
        self,
        scale_min: float = 1e-10,
        clip_max: Optional[float] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.scale_min = scale_min
        self.clip_max = clip_max

    def compute_scale(
        self, F, data: Tensor, observed_indicator: Tensor  # shapes (N, T, C)
    ) -> Tensor:
        # these will have shape (N, C)
        num_observed = F.sum(observed_indicator, axis=1)
        sum_observed = (data.abs() * observed_indicator).sum(axis=1)

        # first compute a global scale per-dimension
        total_observed = num_observed.sum(axis=0)
        denominator = F.maximum(total_observed, 1.0)
        default_scale = sum_observed.sum(axis=0) / denominator  # shape (C, )

        # then compute a per-item, per-dimension scale
        denominator = F.maximum(num_observed, 1.0)
        scale = sum_observed / denominator  # shape (N, C)

        # use per-batch scale when no element is observed
        # or when the sequence contains only zeros
        cond = F.broadcast_greater(sum_observed, F.zeros_like(sum_observed))
        scale = F.where(
            cond,
            scale,
            F.broadcast_mul(default_scale, F.ones_like(num_observed)),
        )

        return F.maximum(scale, self.scale_min)

    # noinspection PyMethodOverriding
    def hybrid_forward(
        self,
        F,
        data: Tensor,
        observed_indicator: Tensor,
        scale: Optional[Tensor],
        rep_params: List[Tensor],
        **kwargs,
    ) -> Tuple[Tensor, Tensor, List[Tensor]]:
        data = F.cast(data, dtype="float32")

        if scale is None:
            scale = self.compute_scale(F, data, observed_indicator)
            scale = scale.expand_dims(axis=1)

        scaled_data = F.broadcast_div(data, scale)

        if self.clip_max is not None:
            scaled_data = F.clip(scaled_data, -self.clip_max, self.clip_max)

        return scaled_data, scale, []

    def post_transform(
        self, F, samples: Tensor, scale: Tensor, rep_params: List[Tensor]
    ) -> Tensor:
        transf_samples = F.broadcast_mul(samples, scale)
        return transf_samples
