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

# Standard library imports
from typing import Tuple, List

# Third-party imports
import numpy as np

# First-party imports
from gluonts.core.component import validated
from gluonts.model.common import Tensor

# Relative imports
from .distribution import (
    Distribution,
    _sample_multiple,
    getF,
)
from .distribution_output import DistributionOutput


class Categorical(Distribution):
    r"""
    A categorical distribution over num_cats-many categories.

    Parameters
    ----------
    log_probs
        Tensor containing log probabilities of the individual categories, of shape `(*batch_shape, num_cats)`.
    F
    """

    @validated()
    def __init__(self, log_probs: Tensor, F=None) -> None:
        super().__init__()
        self.log_probs = log_probs
        self.num_cats = self.log_probs.shape[-1]
        self.F = F if F else getF(log_probs)
        self.cats = self.F.arange(self.num_cats)
        self._probs = None

    @property
    def probs(self):
        if self._probs is None:
            self._probs = self.log_probs.exp()
        return self._probs

    @property
    def batch_shape(self) -> Tuple:
        return self.log_probs.shape[:-1]

    @property
    def event_shape(self) -> Tuple:
        return ()

    @property
    def event_dim(self) -> int:
        return 0

    @property
    def mean(self):
        return (self.probs * self.cats).sum(axis=-1)

    @property
    def stddev(self):
        ex2 = (self.probs * self.cats.square()).sum(axis=-1)
        return (ex2 - self.mean.square()).sqrt()

    def log_prob(self, x):
        F = self.F
        mask = F.one_hot(x, self.num_cats)
        log_prob = F.broadcast_mul(self.log_probs, mask).sum(axis=-1)
        return log_prob

    def sample(self, num_samples=None, dtype=np.int32):
        def s(bin_probs):
            F = self.F
            indices = F.sample_multinomial(bin_probs)
            return indices

        return _sample_multiple(s, self.probs, num_samples=num_samples).astype(
            "int32"
        )

    @property
    def args(self) -> List:
        return [self.log_probs]


class CategoricalOutput(DistributionOutput):
    distr_cls: type = Categorical

    @validated()
    def __init__(self, num_cats: int) -> None:
        super().__init__()
        assert num_cats > 1, "Number of categories must be larger than one."
        self.args_dim = {"num_cats": num_cats}
        self.distr_cls = Categorical
        self.num_cats = num_cats

    def domain_map(self, F, probs):
        log_probs_s = F.log_softmax(probs)
        return log_probs_s

    def distribution(
        self, distr_args, loc=None, scale=None, **kwargs
    ) -> Distribution:
        distr = Categorical(distr_args)
        return distr

    @property
    def event_shape(self) -> Tuple:
        return ()
