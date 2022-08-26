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

from typing import Dict, Optional, Tuple, cast

import torch
from torch.distributions import constraints

from gluonts.core.component import validated
from gluonts.torch.distributions import BinnedUniforms, GeneralizedPareto

from .distribution_output import DistributionOutput


class SplicedBinnedPareto(BinnedUniforms):
    r"""
    Spliced Binned-Pareto univariate distribution.

    Arguments
    ----------
        bins_lower_bound: The lower bound of the bin edges
        bins_upper_bound: The upper bound of the bin edges
        numb_bins: The number of equidistance bins to allocate between
            `bins_lower_bound` and `bins_upper_bound`. Default value is 100.
        tail_percentile_gen_pareto: The percentile of the distribution that is
            each tail. Default value is 0.05. NB: This symmetric percentile
            can still represent asymmetric upper and lower tails.
    """
    arg_constraints = {
        "logits": constraints.real,
        "lower_gp_xi": constraints.positive,
        "lower_gp_beta": constraints.positive,
        "upper_gp_xi": constraints.positive,
        "upper_gp_beta": constraints.positive,
    }
    support = constraints.real
    has_rsample = False

    def __init__(
        self,
        bins_lower_bound: float,
        bins_upper_bound: float,
        logits: torch.tensor,
        upper_gp_xi: torch.tensor,
        upper_gp_beta: torch.tensor,
        lower_gp_xi: torch.tensor,
        lower_gp_beta: torch.tensor,
        numb_bins: int = 100,
        tail_percentile_gen_pareto: float = 0.05,
        validate_args=None,
    ):
        assert (
            tail_percentile_gen_pareto > 0 and tail_percentile_gen_pareto < 0.5
        ), "tail_percentile_gen_pareto must be between (0,1)"
        self.tail_percentile_gen_pareto = torch.tensor(
            tail_percentile_gen_pareto
        )

        device = logits.device
        self.tail_percentile_gen_pareto = self.tail_percentile_gen_pareto.to(
            device
        )

        self.lower_gp_xi = lower_gp_xi
        self.lower_gp_beta = lower_gp_beta
        self.lower_gen_pareto = GeneralizedPareto(
            self.lower_gp_xi, self.lower_gp_beta
        )

        self.upper_gp_xi = upper_gp_xi
        self.upper_gp_beta = upper_gp_beta
        self.upper_gen_pareto = GeneralizedPareto(
            self.upper_gp_xi, self.upper_gp_beta
        )

        setattr(self, "lower_gp_xi", self.lower_gp_xi)
        setattr(self, "lower_gp_beta", self.lower_gp_beta)
        setattr(self, "upper_gp_xi", self.upper_gp_xi)
        setattr(self, "upper_gp_beta", self.upper_gp_beta)

        super(SplicedBinnedPareto, self).__init__(
            bins_lower_bound,
            bins_upper_bound,
            logits,
            numb_bins,
            validate_args,
        )

    # TODO:
    #   - need another implementation of the mean dependent on the tails

    def log_prob(self, x: torch.tensor, for_training=True):
        """
        Arguments
        ----------
        x: a tensor of size 'batch_size', 1
        for_training: boolean to indicate a return of the log-probability, or
            of the loss (which is an adjusted log-probability)
        """

        # Compute upper and lower tail thresholds at current time from
        # their percentiles
        upper_percentile = self._icdf_binned(
            torch.ones_like(x) * (1 - self.tail_percentile_gen_pareto)
        )
        lower_percentile = self._icdf_binned(
            torch.ones_like(x) * self.tail_percentile_gen_pareto
        )
        # upper_percentile.shape: (*batch_shape)
        # lower_percentile.shape: (*batch_shape)

        upper_percentile = upper_percentile.detach()
        lower_percentile = lower_percentile.detach()

        # Log-prob given binned distribution
        logp_bins = self.log_binned_p(x)
        logp = logp_bins.double()
        # logp.shape: (*batch_shape)

        # We obtain the log probabilities under the tail distributions:
        upper_gen_pareto_log_prob = self.upper_gen_pareto.log_prob(
            torch.abs(x.squeeze(dim=-1) - upper_percentile)
        ) + torch.log(self.tail_percentile_gen_pareto)
        lower_gen_pareto_log_prob = self.lower_gen_pareto.log_prob(
            torch.abs(lower_percentile - x.squeeze(dim=-1))
        ) + torch.log(self.tail_percentile_gen_pareto)
        # For the two log prob calls above, we adjust the value so that it
        # corresponds to the value in the tail. We take the absolute value of
        # what we give to the gen pareto because else the gradients are nan.
        # The torch,where select the correct ones and so the values lower than
        # zero are ignored, but the backward pass of pytorch has an issue with
        # nans in where even if they are not selected

        # By default during training we want to optimise the log-prob of both
        # the binned and the gen pareto at the tails
        # if not for training, we want to only have the gen pareto at the tails
        if for_training:
            # Log-prob given upper tail distribution
            logp += torch.where(
                x > upper_percentile,
                upper_gen_pareto_log_prob,
                torch.zeros_like(logp),
            )

            # Log-prob given upper tail distribution
            logp += torch.where(
                x < lower_percentile,
                lower_gen_pareto_log_prob,
                torch.zeros_like(logp),
            )
        else:
            # Log-prob given upper tail distribution
            logp = torch.where(
                x > upper_percentile,
                upper_gen_pareto_log_prob,
                logp,
            )

            # Log-prob given upper tail distribution
            logp = torch.where(
                x < lower_percentile,
                lower_gen_pareto_log_prob,
                logp,
            )
        return logp

    def pdf(self, x):
        """
        Probability for a tensor of data points `x`.
        'x' is to have shape (*batch_shape)
        """
        # By default we put the for training parameter of the pdf on false as
        # one tends to train with the log-prob
        return torch.exp(self.log_prob(x, for_training=False))

    def _inverse_cdf(self, quantiles: torch.tensor):
        """
        Inverse cdf of a tensor of quantile `quantiles`
        'quantiles' is of shape (*batch_shape) with values between (0.0, 1.0)
        """

        # The quantiles for the body of the distribution:
        icdf_body = self._icdf_binned(quantiles)

        # The quantiles if they are in the lower tail:
        adjusted_percentile_for_lower = 1 - (
            quantiles / self.tail_percentile_gen_pareto
        )
        icdf_lower = self._icdf_binned(
            torch.ones_like(quantiles) * self.tail_percentile_gen_pareto
        ) - self.lower_gen_pareto.icdf(adjusted_percentile_for_lower)

        # The quantiles if they are in the upper tail:
        adjusted_percentile_for_upper = (
            quantiles - (1.0 - self.tail_percentile_gen_pareto)
        ) / self.tail_percentile_gen_pareto
        icdf_upper = self.upper_gen_pareto.icdf(
            adjusted_percentile_for_upper
        ) + self._icdf_binned(
            torch.ones_like(quantiles) * (1 - self.tail_percentile_gen_pareto)
        )

        # Putting them together:
        icdf_value = icdf_body

        icdf_value = torch.where(
            quantiles < self.tail_percentile_gen_pareto, icdf_lower, icdf_value
        )

        icdf_value = torch.where(
            quantiles > 1 - self.tail_percentile_gen_pareto,
            icdf_upper,
            icdf_value,
        )

        return icdf_value

    def cdf(self, x: torch.tensor):
        """
        Cumulative density tensor for a tensor of data points `x`.
        'x' is expected to be of shape (*batch_shape)
        """
        for i in range(0, len(x.shape)):
            assert (
                x.shape[i] == self.batch_shape[i]
            ), "We expect the input to be a tensor of size batch_shape"

        lower_percentile_value = self.icdf(self.tail_percentile_gen_pareto)
        upper_percentile_value = self.icdf(1 - self.tail_percentile_gen_pareto)

        # The cdf of the main distribution body:
        cdf_body = self._cdf_binned(x)

        # The cdf for the lower tail:
        adjusted_x_for_lower = lower_percentile_value - x
        cdf_lower = (
            1.0 - self.lower_gen_pareto.cdf(adjusted_x_for_lower)
        ) * self.tail_percentile_gen_pareto

        # The cdf for the upper tail:
        adjusted_x_for_upper = x - upper_percentile_value
        cdf_upper = self.upper_gen_pareto.cdf(
            adjusted_x_for_upper
        ) * self.tail_percentile_gen_pareto + (
            1 - self.tail_percentile_gen_pareto
        )

        # Putting them together:
        cdf_value = cdf_body

        cdf_value = torch.where(
            x < lower_percentile_value, cdf_lower, cdf_value
        )

        cdf_value = torch.where(
            x > upper_percentile_value,
            cdf_upper,
            cdf_value,
        )
        return cdf_value


class SplicedBinnedParetoOutput(DistributionOutput):
    distr_cls: type = SplicedBinnedPareto

    @validated()
    def __init__(
        self,
        bins_lower_bound: float,
        bins_upper_bound: float,
        num_bins: int,
        tail_percentile_gen_pareto: float,
    ) -> None:
        super().__init__(self)

        assert (
            tail_percentile_gen_pareto > 0 and tail_percentile_gen_pareto < 0.5
        ), "tail_percentile_gen_pareto must be between (0,0.5)"
        assert (
            isinstance(num_bins, int) and num_bins > 1
        ), "num_bins should be an integer and greater than 1"
        assert bins_lower_bound < bins_upper_bound, (
            f"bins_lower_bound {bins_lower_bound} needs to less than "
            f"bins_upper_bound {bins_upper_bound}"
        )

        self.num_bins = num_bins
        self.bins_lower_bound = bins_lower_bound
        self.bins_upper_bound = bins_upper_bound

        self.tail_percentile_gen_pareto = tail_percentile_gen_pareto

        self.args_dim = cast(
            Dict[str, int],
            {
                "logits": num_bins,
                "upper_gp_xi": 1,
                "upper_gp_beta": 1,
                "lower_gp_xi": 1,
                "lower_gp_beta": 1,
            },
        )

    @classmethod
    def domain_map(
        cls,
        logits: torch.Tensor,
        upper_gp_xi: torch.Tensor,
        upper_gp_beta: torch.Tensor,
        lower_gp_xi: torch.Tensor,
        lower_gp_beta: torch.Tensor,
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:

        logits = torch.abs(logits)

        upper_gp_xi = torch.abs(upper_gp_xi)
        upper_gp_beta = torch.abs(upper_gp_beta)
        lower_gp_xi = torch.abs(lower_gp_xi)
        lower_gp_beta = torch.abs(lower_gp_beta)

        return logits, upper_gp_xi, upper_gp_beta, lower_gp_xi, lower_gp_beta

    def distribution(
        self,
        distr_args,
        loc: Optional[torch.Tensor] = 0,
        scale: Optional[torch.Tensor] = None,
    ) -> BinnedUniforms:
        return self.distr_cls(
            self.bins_lower_bound,
            self.bins_upper_bound,
            *distr_args,
            self.num_bins,
            self.tail_percentile_gen_pareto,
        )

    @property
    def event_shape(self) -> Tuple:
        return ()
