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
import torch.nn.functional as F
from torch.distributions import Distribution, constraints

from gluonts.core.component import validated

from .distribution_output import DistributionOutput


class BinnedUniforms(Distribution):
    r"""
    Binned uniforms distribution.

    Args:
        bins_lower_bound (float): The lower bound of the bin edges
        bins_upper_bound (float): The upper bound of the bin edges
        numb_bins (int): The number of equidistance bins to allocate between
            `bins_lower_bound` and `bins_upper_bound`. Default value is 100.
        logits (tensor): the logits defining the probability of each bins.
            These are softmaxed. The tensor is of shape (*batch_shape,)
        validate_args (bool) from the pytorch Distribution class
    """
    arg_constraints = {"logits": constraints.real}
    support = constraints.real
    has_rsample = False

    def __init__(
        self,
        bins_lower_bound: float,
        bins_upper_bound: float,
        logits: torch.tensor,
        numb_bins: int = 100,
        validate_args: bool = None,
    ):
        assert bins_lower_bound < bins_upper_bound, (
            f"bins_lower_bound {bins_lower_bound} needs to less than "
            f"bins_upper_bound {bins_upper_bound}"
        )
        assert (
            logits.shape[-1] == numb_bins
        ), "The distribution requires one logit per bin."

        self.logits = logits
        setattr(self, "logits", self.logits)

        device = logits.device

        super(BinnedUniforms, self).__init__(
            batch_shape=logits.shape[:-1],
            event_shape=logits.shape[-1],
            validate_args=validate_args,
        )

        self.numb_bins = numb_bins

        # Creation the bin locations
        # Bins locations are placed uniformly between bins_lower_bound and
        # bins_upper_bound, though more complex methods could be used
        self.bin_min = bins_lower_bound
        self.bin_max = bins_upper_bound

        self.bin_edges = torch.linspace(
            self.bin_min, self.bin_max, self.numb_bins + 1
        )
        self.bin_widths = self.bin_edges[1:] - self.bin_edges[:-1]
        self.bin_centres = (self.bin_edges[1:] + self.bin_edges[:-1]) * 0.5

        self.bin_edges = self.bin_edges.to(device)
        self.bin_widths = self.bin_widths.to(device)
        self.bin_centres = self.bin_centres.to(device)

    @property
    def mode(self):
        """
        Returns the mode of the distribution.

        mode.shape : (*batch_shape,)
        """
        bins_prob = self.bins_prob
        values_max, index_max = torch.max(bins_prob, dim=-1)

        indicator_max = values_max.unsqueeze(dim=-1) == bins_prob

        # print(indicator_max.shape)
        bin_centres = self.bin_centres.unsqueeze(dim=0)
        # upper_edges.shape: (1, numb_bins)
        batch_shape_extended = self.bins_prob[..., 0:1].shape
        bin_centres = bin_centres.repeat(batch_shape_extended)

        mode = (bin_centres * indicator_max).sum(dim=-1)

        return mode

    @property
    def median(self):
        """
        Returns the median of the distribution.

        median.shape : (*batch_shape,)
        """
        return self.icdf(torch.tensor(0.5))

    @property
    def mean(self):
        """
        Returns the mean of the distribution.

        mean.shape : (*batch_shape,)
        """
        batch_shape_extended = self.bins_prob[..., 0:1].shape
        bin_centres = self.bin_centres.unsqueeze(dim=0)
        # bin_centres.shape: (1, numb_bins)
        bin_centres = bin_centres.repeat(batch_shape_extended)
        # bin_centres.shape: (*batch_shape, numb_bins)
        return torch.mean(bin_centres * self.bins_prob, dim=-1)

    @property
    def bins_prob(self):
        """
        Returns the probability of the observed point to be in each of the bins
        bins_prob.shape: (*batch_shape, event_shape).
        event_shape is numb_bins
        """
        bins_prob = self.log_bins_prob.exp()
        return bins_prob

    @property
    def log_bins_prob(self):
        return F.log_softmax(self.logits, dim=-1)  # log_softmax along bins

    def log_prob(self, x):
        """
        Log probability for a tensor of datapoints `x`.
        'x' is to have shape (*batch_shape)
        """
        for i in range(0, len(x.shape)):
            assert (
                x.shape[i] == self.batch_shape[i]
            ), "We expect the input to be a tensor of size batch_shape"
        return self.log_binned_p(x)

    def log_binned_p(self, x):
        """
        Log probability for a tensor of datapoints `x`.
        'x' is to have shape (*batch_shape)

        """
        one_hot_bin_indicator = self.get_one_hot_bin_indicator(
            x, in_float=True
        )
        # one_hot_bin_indicator.shape: (*batch_shape, numb_bins)
        logp = (one_hot_bin_indicator * self.log_bins_prob).sum(dim=-1)
        # logp.shape: (*batch_shape)
        return logp

    def pdf(self, x):
        """
        Probability for a tensor of data points `x`.
        'x' is to have shape (*batch_shape)
        """
        return torch.exp(self.log_prob(x))

    def get_one_hot_bin_indicator(self, x, in_float=False):
        """
        'x' is to have shape (*batch_shape) which can be for example () or
        (32, ) or (32, 168, )
        """
        for i in range(0, len(x.shape)):
            assert (
                x.shape[i] == self.batch_shape[i]
            ), "We expect the input to be a tensor of size batch_shape"

        numb_dim_batch_shape = len(x.shape)

        x_copy = x
        x = x.unsqueeze(dim=-1)
        # x.shape: (*batch_shape, 1)

        upper_edges = self.bin_edges[1:]
        for i in range(0, numb_dim_batch_shape):
            upper_edges = upper_edges.unsqueeze(dim=0)
        # upper_edge.shape: [1, ... ,numb_bins]

        lower_edge = self.bin_edges[:-1]
        for i in range(0, numb_dim_batch_shape):
            lower_edge = lower_edge.unsqueeze(dim=0)
        # lower_edge.shape: [1, ... ,numb_bins]

        one_hot_bin_indicator = ((lower_edge <= x) * (x < upper_edges)).long()
        # one_hot_bin_indicator.shape: [*batch_shape, numb_bins]

        # This handles if x falls outside of [self.bin_min, self.bin_max]
        is_higher_than_last_edge = x_copy >= self.bin_edges[..., -1]
        # is_higher_than_last_edge: [*batch_shape, numb_dim]
        is_lower_than_first_edge = x_copy <= self.bin_edges[..., 0]
        # is_lower_than_first_edge: [*batch_shape, numb_dim]

        one_hot_bin_indicator[..., -1][is_higher_than_last_edge] = 1
        one_hot_bin_indicator[..., 0][is_lower_than_first_edge] = 1

        if not in_float:
            return one_hot_bin_indicator == 1  # booleans
        else:
            return one_hot_bin_indicator.float()  # floats

    def icdf(self, quantiles):
        """
        Inverse cdf of a tensor of quantile `quantiles`
        'quantiles' is of shape (*batch_shape) with values between (0.0, 1.0)

        This is the function to be called from the outside.
        """
        assert (quantiles >= 0.0).all(), "quantiles must be between (0.0, 1.0)"
        assert (quantiles <= 1.0).all(), "quantiles must be between (0.0, 1.0)"

        # If given a single value as quantile, we put it to batch size
        if (
            len(quantiles.shape) == 0
            or len(quantiles.shape) == 1
            and quantiles.shape[0] == 1
        ):
            batch_shape = self.bins_prob[..., 0].shape
            quantiles = quantiles.repeat(batch_shape)

        for i in range(0, len(quantiles.shape)):
            assert quantiles.shape[i] == self.batch_shape[i], (
                "We expect the quantile to be either a single float or a "
                "tensor of size batch_shape"
            )

        return self._inverse_cdf(quantiles)

    def _inverse_cdf(self, quantiles):
        """
        Inverse cdf of a tensor of quantile `quantiles`
        'quantiles' is of shape (*batch_shape) with values between (0.0, 1.0)
        """
        return self._icdf_binned(quantiles)

    def _icdf_binned(self, quantiles):
        """
        Inverse cdf of a tensor of quantile `quantiles`
        'quantiles' is of shape (*batch_shape) with values between (0.0, 1.0)
        """
        quantiles = quantiles.unsqueeze(dim=-1)
        # quantiles.shape: (*batch_shape, 1)

        batch_shape_extended = quantiles.shape

        bins_prob = self.bins_prob

        # For each bin we get the cdf up to the bin (lower) and the cdf
        # including the bin (upper)
        incomplete_cdf_upper = bins_prob.cumsum(dim=-1)
        # incomplete_cdf_upper.shape: (*batch_shape, numb_bins)
        incomplete_cdf_lower = torch.zeros_like(incomplete_cdf_upper)
        incomplete_cdf_lower[..., 1:] = incomplete_cdf_upper[..., :-1]
        # incomplete_cdf_lower.shape: (*batch_shape, numb_bins)

        one_hot_bin_indicator = (incomplete_cdf_lower <= quantiles) * (
            quantiles < incomplete_cdf_upper
        )
        # one_hot_bin_indicator.shape: (*batch_shape, numb_bins)
        # Handling the quantile equal to 1.0
        higher_than_last = quantiles[..., 0] >= incomplete_cdf_upper[..., -1]
        one_hot_bin_indicator[..., -1][higher_than_last] = True

        upper_edges = self.bin_edges[1:].unsqueeze(dim=0)
        # upper_edges.shape: (1, numb_bins)
        upper_edges = upper_edges.repeat(batch_shape_extended)
        # upper_edges.shape: (*batch_shape, numb_bins)
        lower_edges = self.bin_edges[:-1].unsqueeze(dim=0)
        # lower_edges.shape: (1, numb_bins)
        lower_edges = lower_edges.repeat(batch_shape_extended)
        # lower_edges.shape: (*batch_shape, numb_bins)

        bin_width = upper_edges[one_hot_bin_indicator].view(
            batch_shape_extended
        ) - lower_edges[one_hot_bin_indicator].view(batch_shape_extended)
        # bin_width.shape: (*batch_shape)

        prob_bin = bins_prob[one_hot_bin_indicator].view(batch_shape_extended)
        # prob_bin.shape: (*batch_shape)

        prob_left = quantiles.view(
            batch_shape_extended
        ) - incomplete_cdf_lower[one_hot_bin_indicator].view(
            batch_shape_extended
        )
        # prob_left.shape: (*batch_shape)
        bin_lower_edge = lower_edges[one_hot_bin_indicator].view(
            batch_shape_extended
        )
        # bin_lower_edge.shape: (*batch_shape)

        result_icdf = bin_width * prob_left / prob_bin + bin_lower_edge
        return result_icdf.squeeze(dim=-1)

    def cdf(self, x):
        """
        Cumulative density tensor for a tensor of data points `x`.
        'x' is expected to be of shape (*batch_shape)
        """
        for i in range(0, len(x.shape)):
            assert (
                x.shape[i] == self.batch_shape[i]
            ), "We expect the input to be a tensor of size batch_shape"
        return self._cdf_binned(x)

    def _cdf_binned(self, x):
        """
        Cumulative density tensor for a tensor of data points `x`.
        'x' is expected to be of shape (*batch_shape)

        The cdf is composed of 2 parts:
            the cdf up to the bin
            the cdf within the bin that the point falls into (modeled with a
            uniform distribution)
        """

        bins_prob = self.bins_prob

        batch_shape_extended = bins_prob[..., 0:1].shape

        # Get the location of points in the bins
        one_hot_bin_indicator = self.get_one_hot_bin_indicator(x)
        # one_hot_bin_indicator.shape: (*batch_shape, numb_bins)

        # Get the cdf over the bins i.e. the probability mass up to each
        # bin's upper edge
        incomplete_cdf = bins_prob.cumsum(dim=-1) - bins_prob
        # incomplete_cdf.shape: (*batch_shape,numb_bins)

        cdf_up_to_bin = (
            (incomplete_cdf * one_hot_bin_indicator)
            .sum(dim=-1)
            .unsqueeze(dim=-1)
        )
        # incomplete_cdf.shape: (*batch_shape,1)

        # Prepare to select the edges of the bins that the points fall into
        upper_edges = self.bin_edges[1:].unsqueeze(dim=0)
        # upper_edges.shape: (1...,numb_bins)
        upper_edges = upper_edges.repeat(batch_shape_extended)
        # upper_edges.shape: (*batch_shape,numb_bins)
        lower_edges = self.bin_edges[:-1].unsqueeze(dim=0)
        lower_edges = lower_edges.repeat(batch_shape_extended)
        # lower_edges.shape: (*batch_shape,numb_bins)

        # With the edges and the point value we can get the cdf within the
        # bin given that they are uniform
        # distributions, and weight it by the probability of the bin
        bin_width = upper_edges[one_hot_bin_indicator].view(
            batch_shape_extended
        ) - lower_edges[one_hot_bin_indicator].view(batch_shape_extended)
        # bin_width.shape: (*batch_shape,1)

        dist_in_bin = x.unsqueeze(dim=-1) - lower_edges[
            one_hot_bin_indicator
        ].view(batch_shape_extended)
        # dist_in_bin.shape: (*batch_shape,1)
        dist_in_bin = torch.max(
            torch.min(dist_in_bin, bin_width), torch.zeros_like(dist_in_bin)
        )
        # this is for points falling outside the bins

        cdf_in_bin = (
            bins_prob[one_hot_bin_indicator].view(batch_shape_extended)
            * dist_in_bin
            / bin_width
        )
        # cdf_in_bin.shape: (*batch_shape,1)

        return (
            (cdf_in_bin + cdf_up_to_bin)
            .reshape(batch_shape_extended)
            .squeeze(dim=-1)
        )

    def sample(self, sample_shape=torch.Size()):
        """
        Returns samples from the distribution.

        Returns:
            samples of shape (*sample_shape, *batch_shape)

        """
        if len(sample_shape) == 0:
            quantiles = torch.rand(self.batch_shape)
            samples = self.icdf(quantiles)
        else:
            samples = torch.zeros(list(sample_shape) + list(self.batch_shape))
            for i in range(sample_shape[0]):
                quantiles = torch.rand(self.batch_shape)
                samples_i = self.icdf(quantiles)
                samples[i, ...] = samples_i

        return samples

    def expand(self, batch_shape, _instance=None):
        raise NotImplementedError

    def variance(self):
        raise NotImplementedError

    def rsample(self, sample_shape=torch.Size()):
        """
        We do not have an implementation for the reparameterization trick yet.
        """
        raise NotImplementedError

    def entropy(self):
        """
        We do not have an implementation of the entropy yet.
        """
        raise NotImplementedError

    def enumerate_support(self, expand=True):
        """
        This is a real valued distribution.
        """
        raise NotImplementedError


class BinnedUniformsOutput(DistributionOutput):
    distr_cls: type = BinnedUniforms

    @validated()
    def __init__(
        self,
        bins_lower_bound: float,
        bins_upper_bound: float,
        num_bins: int,
    ) -> None:
        super().__init__(self)

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

        self.args_dim = cast(
            Dict[str, int],
            {"logits": num_bins},
        )

    @classmethod
    def domain_map(cls, logits: torch.Tensor) -> torch.Tensor:

        logits = torch.abs(logits)

        return logits

    def distribution(
        self,
        distr_args,
        loc: Optional[torch.Tensor] = 0,
        scale: Optional[torch.Tensor] = None,
    ) -> BinnedUniforms:
        return self.distr_cls(
            self.bins_lower_bound,
            self.bins_upper_bound,
            distr_args,
            self.num_bins,
        )

    @property
    def event_shape(self) -> Tuple:
        return ()
