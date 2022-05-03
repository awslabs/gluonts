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

from typing import Optional

import numpy as np

import torch
import torch.nn
import torch.nn.functional as F
import torch.optim

from .genpareto import GenPareto


class Binned(torch.nn.Module):
    r"""
    Binned univariate distribution designed as an nn.Module

    Arguments
    ----------
    bins_lower_bound: The lower bound of the bin edges
    bins_upper_bound: The upper bound of the bin edges
    nbins: The number of equidistance bins to allocate between `bins_lower_bound` and `bins_upper_bound`. Default value is 100.
    smoothing_indicator: The method of smoothing to perform on the bin probabilities
    """

    def __init__(
        self,
        bins_lower_bound: float,
        bins_upper_bound: float,
        nbins: int = 100,
        smoothing_indicator: Optional[str] = [None, "cheap", "kernel"][1],
        validate_args=None,
    ):
        super().__init__()

        assert (
            bins_lower_bound.shape.numel() == 1
        ), "bins_lower_bound needs to have shape torch.Size([1])"
        assert (
            bins_upper_bound.shape.numel() == 1
        ), "bins_upper_bound needs to have shape torch.Size([1])"
        assert bins_lower_bound < bins_upper_bound, (
            f"bins_lower_bound {bins_lower_bound} needs to less than"
            f" bins_upper_bound {bins_upper_bound}"
        )

        self.nbins = nbins
        self.epsilon = np.finfo(np.float32).eps
        self.smooth_indicator = smoothing_indicator

        # Creation the bin locations
        # Bins locations are placed uniformly between bins_lower_bound and bins_upper_bound, though more complex methods could be used
        self.bin_min = bins_lower_bound - self.epsilon * 6
        self.bin_max = bins_upper_bound + self.epsilon * 6
        self.bin_edges = torch.linspace(self.bin_min, self.bin_max, nbins + 1)
        self.bin_widths = self.bin_edges[1:] - self.bin_edges[:-1]
        self.bin_centres = (self.bin_edges[1:] + self.bin_edges[:-1]) * 0.5

        logits = torch.ones(nbins)
        logits = (
            logits / logits.sum() / (1 + self.epsilon) / self.bin_widths.mean()
        )
        self.logits = torch.log(logits)

        # Keeps track of mini-batches
        self.idx = None

        self.device = None

    def to_device(self, device):
        """
        Moves members to a specified torch.device.
        """
        self.device = device
        self.bin_min = self.bin_min.to(device)
        self.bin_max = self.bin_max.to(device)
        self.bin_edges = self.bin_edges.to(device)
        self.bin_widths = self.bin_widths.to(device)
        self.bin_centres = self.bin_centres.to(device)

    def forward(self, x):
        """
        Takes input x as new logits.
        """
        self.logits = x
        return self.logits

    def log_bins_prob(self):
        if self.idx is None:
            log_bins_prob = F.log_softmax(self.logits, dim=0).sub(
                torch.log(self.bin_widths)
            )
        else:
            log_bins_prob = F.log_softmax(self.logits[self.idx, :], dim=0).sub(
                torch.log(self.bin_widths)
            )
        return log_bins_prob.float()

    def bins_prob(self):
        bins_prob = self.log_bins_prob().exp()
        return bins_prob

    def bins_cdf(self):
        incomplete_cdf = self.bins_prob().mul(self.bin_widths).cumsum(dim=0)
        zero = 0 * incomplete_cdf[0].view(1)  # ensured to be on same device
        return torch.cat((zero, incomplete_cdf))

    def log_binned_p(self, xx):
        """
        Log probability for one datapoint.
        """
        assert xx.shape.numel() == 1, "log_binned_p() expects univariate"

        # Transform xx in to a one-hot encoded vector to get bin location
        vect_above = xx - self.bin_edges[1:]
        vect_below = self.bin_edges[:-1] - xx
        one_hot_bin_indicator = (vect_above * vect_below >= 0).float()

        if xx > self.bin_edges[-1]:
            one_hot_bin_indicator[-1] = 1.0
        elif xx < self.bin_edges[0]:
            one_hot_bin_indicator[0] = 1.0
        if not (one_hot_bin_indicator == 1).sum() == 1:
            print(
                f"Warning in log_p(self, xx): for xx={xx.item()},"
                " one_hot_bin_indicator value_counts are"
                f" {one_hot_bin_indicator.unique(return_counts=True)}"
            )

        if self.smooth_indicator == "kernel":
            # The kernel variant is better but slows down training quite a bit
            idx_one_hot = torch.argmax(one_hot_bin_indicator)
            kernel = [0.006, 0.061, 0.242, 0.383, 0.242, 0.061, 0.006]
            len_kernel = len(kernel)
            for i in range(len_kernel):
                idx = i - len_kernel // 2 + idx_one_hot
                if idx in range(len(one_hot_bin_indicator)):
                    one_hot_bin_indicator[idx] = kernel[i]

        elif self.smooth_indicator == "cheap":
            # This variant is cheaper in computation time
            idx_one_hot = torch.argmax(one_hot_bin_indicator)
            if not idx_one_hot + 1 >= len(one_hot_bin_indicator):
                one_hot_bin_indicator[idx_one_hot + 1] = 0.5
            if not idx_one_hot - 1 < 0:
                one_hot_bin_indicator[idx_one_hot - 1] = 0.5
            if not idx_one_hot + 2 >= len(one_hot_bin_indicator):
                one_hot_bin_indicator[idx_one_hot + 2] = 0.25
            if not idx_one_hot - 2 < 0:
                one_hot_bin_indicator[idx_one_hot - 2] = 0.25

        logp = torch.dot(one_hot_bin_indicator, self.log_bins_prob())
        return logp

    def log_p(self, xx):
        """
        Log probability for one datapoint `xx`.
        """
        assert xx.shape.numel() == 1, "log_p() expects univariate"
        return self.log_binned_p(xx)

    def log_prob(self, x):
        """
        Log probability for a tensor of datapoints `x`.
        """
        x = x.view(x.shape.numel())
        self.idx = 0
        if x.shape[0] == 1:
            self.idx = None
        lpx = self.log_p(x[0]).view(1)

        if x.shape.numel() == 1:
            return lpx

        for xx in x[1:]:
            self.idx += 1
            lpxx = self.log_p(xx).view(1)
            lpx = torch.cat((lpx, lpxx), 0)

        self.idx = None
        return lpx

    def cdf_binned_components(
        self, xx, idx=0, cum_density=torch.tensor([0.0])
    ):
        """
        Cumulative density given bins for one datapoint `xx`, where
        `cum_density` is the cdf up to bin_edges `idx`  which must be lower
        than `xx`
        """
        assert xx.shape.numel() == 1, "cdf_components() expects univariate"

        bins_range = self.bin_edges[-1] - self.bin_edges[0]
        bin_cdf_relative = torch.tensor([0.0])
        if idx == 0:
            cum_density = torch.tensor([0.0])

        while xx > self.bin_edges[idx] and idx < self.nbins:
            bin_width = self.bin_edges[idx + 1] - self.bin_edges[idx]
            if xx < self.bin_edges[idx + 1]:
                bin_cdf = torch.distributions.uniform.Uniform(
                    self.bin_edges[idx], self.bin_edges[idx + 1]
                ).cdf(xx)
                bin_cdf_relative = bin_cdf * bin_width / bins_range
                break
            else:
                cum_density += self.bins_prob()[idx] * bin_width
                idx += 1
        return idx, cum_density, bin_cdf_relative

    def cdf_components(self, xx, idx=0, cum_density=torch.tensor([0.0])):
        """
        Cumulative density for one datapoint `xx`, where `cum_density` is the
        cdf up to bin_edges `idx` which must be lower than `xx`
        """
        return self.cdf_binned_components(xx, idx, cum_density)

    def cdf(self, x):
        """
        Cumulative density tensor for a tensor of datapoints `x`.
        """
        x = x.view(x.shape.numel())
        sorted_x = x.sort()
        x, unsorted_index = sorted_x.values, sorted_x.indices

        idx, cum_density, bin_cdf_relative = self.cdf_components(
            x[0], idx=0, cum_density=torch.tensor([0.0])
        )
        cdf_tensor = (cum_density + bin_cdf_relative).view(1)
        if x.shape.numel() == 1:
            return cdf_tensor

        for xx in x[1:]:
            idx, cum_density, bin_cdf_relative = self.cdf_components(
                xx, idx, cum_density
            )
            cdfx = (cum_density + bin_cdf_relative).view(1)
            cdf_tensor = torch.cat((cdf_tensor, cdfx), 0)

        cdf_tensor = cdf_tensor[unsorted_index]
        return cdf_tensor

    def inverse_binned_cdf(self, value):
        """
        Inverse binned cdf of a single quantile `value`
        """
        assert (
            value.shape.numel() == 1
        ), "inverse_binned_cdf() expects univariate"
        if value == 0.0:
            return self.bin_edges[0]
        if value == 1:
            return self.bin_edges[-1]

        vect_above = value - self.bins_cdf()[1:]
        vect_below = self.bins_cdf()[:-1] - value

        if (vect_above == 0).any():
            result = self.bin_edges[1:][vect_above == 0]
        elif (vect_below == 0).any():
            result = self.bin_edges[:-1][vect_below == 0]
        else:
            one_hot_edge_indicator = vect_above * vect_below >= 0  # .float()
            low = self.bin_edges[:-1][one_hot_edge_indicator]
            high = self.bin_edges[1:][one_hot_edge_indicator]
            value_relative = (
                value - self.bins_cdf()[:-1][one_hot_edge_indicator]
            )
            result = torch.distributions.uniform.Uniform(low, high).icdf(
                value_relative
            )

        return result

    def inverse_cdf(self, value):
        """
        Inverse cdf of a single percentile `value`
        """
        return self.inverse_binned_cdf(value)

    def icdf(self, values):
        """
        Inverse cdf of a tensor of quantile `values`
        """
        if self.device is not None:
            values = values.to(self.device)

        values = values.view(values.shape.numel())
        icdf_tensor = self.inverse_cdf(values[0])
        icdf_tensor = icdf_tensor.view(1)

        if values.shape.numel() == 1:
            return icdf_tensor

        for value in values[1:]:
            icdf_value = self.inverse_cdf(value).view(1)
            icdf_tensor = torch.cat((icdf_tensor, icdf_value), 0)

        return icdf_tensor


class SplicedBinnedPareto(Binned):
    r"""
    Spliced Binned-Pareto univariate distribution.

    Arguments
    ----------
    bins_lower_bound: The lower bound of the bin edges
    bins_upper_bound: The upper bound of the bin edges
    nbins: The number of equidistance bins to allocate between `bins_lower_bound` and `bins_upper_bound`. Default value is 100.
    percentile_gen_pareto: The percentile of the distribution that is each tail. Default value is 0.05. NB: This symmetric percentile can still represent asymmetric upper and lower tails.
    """

    def __init__(
        self,
        bins_lower_bound: float,
        bins_upper_bound: float,
        nbins: int = 100,
        percentile_gen_pareto: torch.Tensor = torch.tensor(0.05),
        validate_args=None,
    ):
        super().__init__(
            bins_lower_bound, bins_upper_bound, nbins, validate_args
        )

        assert (
            percentile_gen_pareto > 0 and percentile_gen_pareto < 1
        ), "percentile_gen_pareto must be between (0,1)"
        self.percentile_gen_pareto = percentile_gen_pareto

        self.lower_xi = torch.nn.Parameter(torch.tensor(0.5))
        self.lower_beta = torch.nn.Parameter(torch.tensor(0.5))
        self.lower_gen_pareto = GenPareto(self.lower_xi, self.lower_beta)

        self.upper_xi = torch.nn.Parameter(torch.tensor(0.5))
        self.upper_beta = torch.nn.Parameter(torch.tensor(0.5))
        self.upper_gen_pareto = GenPareto(self.upper_xi, self.upper_beta)

        self.lower_xi_batch = None
        self.lower_beta_batch = None
        self.upper_xi_batch = None
        self.upper_beta_batch = None

    def to_device(self, device):
        """
        Moves members to a specified torch.device.
        """
        self.device = device
        self.bin_min = self.bin_min.to(device)
        self.bin_max = self.bin_max.to(device)
        self.bin_edges = self.bin_edges.to(device)
        self.bin_widths = self.bin_widths.to(device)
        self.bin_centres = self.bin_centres.to(device)
        self.logits = self.logits.to(device)

    def forward(self, x):
        """Takes input x as the new parameters to specify the bin
        probabilities: logits for the base distribution, and xi and beta for
        each tail distribution."""
        if len(x.shape) > 1:
            # If mini-batching
            self.logits = x[:, : self.nbins]

            self.lower_xi_batch = F.softplus(x[:, self.nbins])
            self.lower_beta_batch = F.softplus(x[:, self.nbins + 1])

            self.upper_xi_batch = F.softplus(x[:, self.nbins + 2])
            self.upper_beta_batch = F.softplus(x[:, self.nbins + 3])
        else:
            # If not mini-batching
            self.logits = x[: self.nbins]

            self.lower_xi_batch = F.softplus(x[self.nbins])
            self.lower_beta_batch = F.softplus(x[self.nbins + 1])

            self.upper_xi_batch = F.softplus(x[self.nbins + 2])
            self.upper_beta_batch = F.softplus(x[self.nbins + 3])

            self.upper_gen_pareto.xi = self.upper_xi_batch
            self.upper_gen_pareto.beta = self.upper_beta_batch
            self.lower_gen_pareto.xi = self.lower_xi_batch
            self.lower_gen_pareto.beta = self.lower_beta_batch

        return self.logits

    def log_p(self, xx, for_training=True):
        """
        Arguments
        ----------
        xx: one datapoint
        for_training: boolean to indicate a return of the log-probability, or of the loss (which is an adjusted log-probability)
        """
        assert xx.shape.numel() == 1, "log_p() expects univariate"

        # Compute upper and lower tail thresholds at current time from their percentiiles
        upper_percentile = self.icdf(1 - self.percentile_gen_pareto)
        lower_percentile = self.icdf(self.percentile_gen_pareto)

        # Log-prob given binned distribution
        logp_bins = self.log_binned_p(xx) + torch.log(
            1 - 2 * self.percentile_gen_pareto
        )
        logp = logp_bins

        # Log-prob given upper tail distribution
        if xx > upper_percentile:
            if self.upper_xi_batch is not None:
                # self.upper_gen_pareto.xi = torch.square(self.upper_xi_batch[self.idx])
                # self.upper_gen_pareto.beta = torch.square(self.upper_beta_batch[self.idx])
                self.upper_gen_pareto.xi = self.upper_xi_batch[self.idx]
                self.upper_gen_pareto.beta = self.upper_beta_batch[self.idx]
            logp_gen_pareto = self.upper_gen_pareto.log_prob(
                xx - upper_percentile
            ) + torch.log(self.percentile_gen_pareto)
            logp = logp_gen_pareto
            if for_training:
                logp += logp_bins

        # Log-prob given upper tail distribution
        elif xx < lower_percentile:
            if self.lower_xi_batch is not None:
                # self.lower_gen_pareto.xi = torch.square(self.lower_xi_batch[self.idx])
                # self.lower_gen_pareto.beta = torch.square(self.lower_beta_batch[self.idx])
                self.lower_gen_pareto.xi = self.lower_xi_batch[self.idx]
                self.lower_gen_pareto.beta = self.lower_beta_batch[self.idx]
            logp_gen_pareto = self.lower_gen_pareto.log_prob(
                lower_percentile - xx
            ) + torch.log(self.percentile_gen_pareto)
            logp = logp_gen_pareto
            if for_training:
                logp += logp_bins

        return logp

    def cdf_components(self, xx, idx=0, cum_density=torch.tensor([0.0])):
        """
        Cumulative density for one datapoint `xx`, where `cum_density` is the
        cdf up to bin_edges `idx` which must be lower than `xx`
        """
        bin_cdf_relative = torch.tensor([0.0])
        upper_percentile = self.icdf(1 - self.percentile_gen_pareto)
        lower_percentile = self.icdf(self.percentile_gen_pareto)
        if xx < lower_percentile:
            adjusted_xx = lower_percentile - xx
            cum_density = (
                1.0 - self.lower_gen_pareto.cdf(adjusted_xx)
            ) * self.percentile_gen_pareto
        elif xx <= upper_percentile:
            idx, cum_density, bin_cdf_relative = self.cdf_binned_components(
                xx, idx, cum_density
            )
        else:
            adjusted_xx = xx - upper_percentile
            cum_density = (
                1.0 - self.percentile_gen_pareto
            ) + self.upper_gen_pareto.cdf(
                adjusted_xx
            ) * self.percentile_gen_pareto
        return idx, cum_density, bin_cdf_relative

    def inverse_cdf(self, value):
        """
        Inverse cdf of a single percentile `value`
        """
        assert (
            value >= 0.0 and value <= 1.0
        ), "percentile value must be between 0 and 1 inclusive"

        if value < self.percentile_gen_pareto:
            adjusted_percentile = 1 - (value / self.percentile_gen_pareto)
            icdf_value = self.inverse_binned_cdf(
                self.percentile_gen_pareto
            ) - self.lower_gen_pareto.icdf(adjusted_percentile)
        elif value <= 1 - self.percentile_gen_pareto:
            icdf_value = self.inverse_binned_cdf(value)
        else:
            adjusted_percentile = (
                value - (1.0 - self.percentile_gen_pareto)
            ) / self.percentile_gen_pareto
            icdf_value = self.upper_gen_pareto.icdf(
                adjusted_percentile
            ) + self.inverse_binned_cdf(1 - self.percentile_gen_pareto)

        return icdf_value
