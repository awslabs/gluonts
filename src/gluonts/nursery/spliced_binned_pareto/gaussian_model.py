# Question: license here?

import torch
import torch.nn.functional as F
import torch.nn as nn

from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.normal import Normal


class GaussianModel(nn.Module):
    r"""
    Model to learn a univariate Gaussian distribution.

    Arguments
    ----------
    mu: Mean of the Gaussian distribution
    sigma: Standard deviation of the Gaussian distribution
    device: The torch.device to use, typically cpu or gpu id
    """

    def __init__(self, mu, sigma, device=None):
        super(GaussianModel, self).__init__()
        if device is not None:
            self.device = device
            mu = mu.to(device)
            sigma = sigma.to(device)
        self.mu = mu
        self.sigma = sigma
        self.distr = Normal(self.mu, self.sigma)

    def to_device(self, device):
        """
        Moves members to a specified torch.device
        """
        self.device = device

    def forward(self, x):
        """
        Takes input x as new distribution parameters
        """
        # If mini-batching
        if len(x.shape) > 1:
            self.mu_batch = x[:, 0]
            self.sigma_batch = F.softplus(x[:, 1])

        # If not mini-batching
        else:
            self.mu = x[0]
            self.distr = Normal(self.mu, self.sigma)

        return self.distr

    def log_prob(self, x):
        x = x.view(x.shape.numel())
        if x.shape[0] == 1:
            return self.distr.log_prob(x[0]).view(1)

        log_like_arr = torch.ones_like(x)
        for i in range(len(x)):
            self.mu = self.mu_batch[i]
            self.distr = Normal(self.mu, self.sigma)
            lpxx = self.distr.log_prob(x[i]).view(1)
            log_like_arr[i] = lpxx

        lpx = log_like_arr
        return lpx

    def icdf(self, value):
        return self.distr.icdf(value)
