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

from typing import Tuple

from gluonts.mx import Tensor


class Normalization:
    def __init__(self, clip: int = 10, epsilon: float = 1e-8) -> None:
        self._clip = clip
        self._epsilon = epsilon

    def __call__(
        self, F, t: Tensor, scale: Tensor = None, mean: Tensor = None
    ) -> Tensor:
        raise NotImplementedError

    def scale_loc(
        self,
        F,
        scale: Tensor = None,
        mean: Tensor = None,
    ) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError


class CenteredScale(Normalization):
    def __init__(self, clip: int = 10, epsilon: float = 1e-8) -> None:
        super().__init__(clip, epsilon)

    def __call__(
        self,
        F,
        t: Tensor,
        scale: Tensor = None,
        mean: Tensor = None,
    ) -> Tensor:
        x = F.broadcast_div(t, scale)
        x = F.broadcast_sub(x, F.sign(mean))
        return F.clip(x, -self._clip, self._clip)

    def scale_loc(
        self,
        F,
        scale: Tensor = None,
        mean: Tensor = None,
    ) -> Tuple[Tensor, Tensor]:
        loc = F.broadcast_mul(scale, F.sign(mean))
        return scale, loc


class Standardization(Normalization):  # type: ignore
    def __init__(self, clip: int = 10, epsilon: float = 1e-8) -> None:
        super().__init__(clip, epsilon)  # type: ignore

    def __call__(
        self,
        F,
        t: Tensor,
        scale: Tensor = None,
        mean: Tensor = None,
    ) -> Tensor:
        x = F.broadcast_sub(t, mean)
        std = F.sqrt(scale) + self._epsilon
        x = F.broadcast_div(x, std)
        return F.clip(x, -self._clip, self._clip)

    def scale_loc(
        self,
        F,
        scale: Tensor = None,
        mean: Tensor = None,
    ) -> Tuple[Tensor, Tensor]:
        std = F.sqrt(scale) + self._epsilon
        return std, mean


def get_normalization(normalization: str, **kwargs) -> Normalization:
    if normalization == "centeredscale":
        return CenteredScale(**kwargs)
    elif normalization == "standardization":
        return Standardization(**kwargs)
    else:
        raise NotImplementedError
