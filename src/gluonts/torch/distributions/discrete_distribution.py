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

from numbers import Number
from typing import Optional

import torch


class DiscreteDistribution(torch.distributions.Distribution):
    """
    Implements discrete distribution where the underlying random variable
    takes a value from the finite set `values` with the corresponding
    probabilities.

    Note: `values` can have duplicates in which case the probability mass
    of duplicates is added up.

    A natural loss function, especially given that the new observation does
    not have to be from the finite set `values`, is ranked probability score
    (RPS). For this reason and to be consitent with terminology of other
    models, `log_prob` is implemented as the negative RPS.
    """

    def __init__(
        self,
        values: torch.Tensor,
        probs: torch.Tensor,
        validate_args: Optional[bool] = None,
    ):
        if validate_args:
            total_prob = probs.sum(dim=1)
            assert torch.allclose(total_prob, torch.ones_like(total_prob))

        self.values = values
        self.probs = probs
        self.values_sorted, ix_sorted = torch.sort(values, dim=1)
        self.probs_sorted = probs[
            torch.arange(values.shape[0]).unsqueeze(-1), ix_sorted
        ]
        self.CDF = torch.cumsum(self.probs_sorted, dim=1)

        if isinstance(values, Number) and isinstance(probs, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.values_sorted[:, 0:1].size()
        super().__init__(batch_shape, validate_args=validate_args)

    @staticmethod
    def adjust_probs(values_sorted, probs_sorted):
        """
        Puts probability mass of all duplicate values into one position (last
        index of the duplicate).

        Assumption: `values_sorted` is sorted!

        :param values_sorted:
        :param probs_sorted:
        :return:
        """

        def _adjust_probs_per_element(values_sorted, probs_sorted):
            if torch.sum(torch.diff(values_sorted) == 0) == 0:
                # This batch element does not have duplicates.
                probs_adjusted = probs_sorted
            else:
                _, counts = torch.unique_consecutive(
                    values_sorted, return_counts=True
                )

                # list is fine here as it operates on the network inputs
                # (values) not parameters
                unique_splits = torch.split(probs_sorted, list(counts))
                probs_cumsum_per_split = torch.cat(
                    [torch.cumsum(s, dim=0) for s in unique_splits]
                )

                # Puts 0 on the positions where the duplicates occur except
                # for the last position of the duplicate.
                # To have a 1 at the end, we append a value larger than the
                # observed values before calling diff.
                mask_unique_prob = (
                    torch.diff(values_sorted, append=values_sorted[-1:] + 1.0)
                    > 0
                )
                probs_adjusted = probs_cumsum_per_split * mask_unique_prob

            return probs_adjusted

        # Some batch elements have duplicate values, so adjust the
        # corresponding probabilities
        probs_adjusted_it = map(
            _adjust_probs_per_element,
            torch.unbind(values_sorted, dim=0),
            torch.unbind(probs_sorted, dim=0),
        )
        return torch.stack(list(probs_adjusted_it))

    def mean(self):
        return (self.probs_sorted * self.values_sorted).sum(dim=1)

    def log_prob(self, obs: torch.Tensor):
        return -self.rps(obs)

    def rps(self, obs: torch.Tensor, check_for_duplicates: bool = True):
        """
        Implements ranked probability score which is the sum of the qunatile
        losses for all possible quantiles.

        Here, the number of quantiles is finite and is equal to the number of
        unique values in (each batch element of) `obs`.

        Parameters
        ----------
        obs
        check_for_duplicates

        Returns
        -------

        """
        if self._validate_args:
            self._validate_sample(obs)

        probs_sorted, values_sorted = self.probs_sorted, self.values_sorted
        if (
            check_for_duplicates
            and torch.sum(torch.diff(values_sorted) == 0) > 0
        ):
            probs_sorted = self.adjust_probs(
                values_sorted=values_sorted, probs_sorted=probs_sorted
            )

        # Recompute CDF with adjusted probabilities.
        CDF = torch.cumsum(probs_sorted, dim=1)

        mask_non_zero_prob = (probs_sorted > 0).int()
        quantile_losses = mask_non_zero_prob * self.quantile_losses(
            obs,
            quantiles=values_sorted,
            levels=CDF,
        )

        # We return normalized RPS
        return torch.sum(quantile_losses, dim=-1) / torch.sum(
            mask_non_zero_prob, dim=-1
        )

    def quantile_losses(
        self, obs: torch.Tensor, quantiles: torch.Tensor, levels: torch.Tensor
    ):
        assert obs.shape[:-1] == quantiles.shape[:-1]
        assert obs.shape[:-1] == levels.shape[:-1]
        assert obs.shape[-1] == 1

        return torch.where(
            obs >= quantiles,
            levels * (obs - quantiles),
            (1 - levels) * (quantiles - obs),
        )

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        with torch.no_grad():
            cat_distr = torch.distributions.Categorical(
                probs=self.probs_sorted
            )
            samples_ix = cat_distr.sample().reshape(shape=shape)
            samples = self.values_sorted[
                torch.arange(samples_ix.shape[0]).unsqueeze(-1), samples_ix
            ]
            assert samples.shape == self.batch_shape
            return samples
