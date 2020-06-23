import torch

from torch_extensions.distributions.parametrised_distribution import \
    ParametrisedMultivariateNormal
from utils.utils import make_inv_tril_parametrization


class StatePriorModelNoInputs(ParametrisedMultivariateNormal):
    def __init__(self, config):
        covariance_matrix = (config.state_prior_scale ** 2) * torch.eye(
            config.dims.state)
        LVinv_tril, LVinv_logdiag = make_inv_tril_parametrization(
            covariance_matrix)
        super().__init__(
            m=torch.ones(config.dims.state) * config.state_prior_loc,
            LVinv_tril=LVinv_tril,
            LVinv_logdiag=LVinv_logdiag,
            requires_grad_m=True,
            requires_diag_LVinv_tril=False,
            requires_diag_LVinv_logdiag=True,
        )


class StatePriorModeFixedlNoInputs(ParametrisedMultivariateNormal):
    def __init__(self, config):
        covariance_matrix = (config.state_prior_scale ** 2) * torch.eye(
            config.dims.state)
        LVinv_tril, LVinv_logdiag = make_inv_tril_parametrization(
            covariance_matrix)
        super().__init__(
            m=torch.ones(config.dims.state) * config.state_prior_loc,
            LVinv_tril=LVinv_tril,
            LVinv_logdiag=LVinv_logdiag,
            requires_grad_m=False,
            requires_diag_LVinv_tril=False,
            requires_diag_LVinv_logdiag=False,
        )
