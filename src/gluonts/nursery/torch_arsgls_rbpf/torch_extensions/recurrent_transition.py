import torch
from torch import nn
from torch.distributions import MultivariateNormal

from inference.analytical_gausian_linear.inference_step import (
    filter_forward_prediction_step,
)
from torch_extensions.ops import matvec
from torch_extensions.ops import cholesky


class GaussianRecurrentTransition(nn.Module):
    def __init__(
        self,
        conditional_dist_tranform: nn.Module,
        n_state: int,
        n_switch: int,
        is_recurrent: bool = True,
    ):
        super().__init__()
        self.conditional_dist_transform = conditional_dist_tranform
        if is_recurrent:
            self.F = nn.Parameter(
                # torch.nn.init.xavier_normal(torch.empty(n_switch, n_state)) / 4.0,
                torch.nn.init.orthogonal_(torch.empty(n_switch, n_state))
                / 4.0,
                requires_grad=True,
            )
        else:
            self.register_parameter("F", None)

    def forward(self, u, s, x=None, m=None, V=None):
        """
        m and V are from the Gaussian state x_{t-1}.
        Forward marginalises out the Gaussian if m and V given,
        or transforms a sample x if that is given instead.
        """
        # assert len({(x is None), (m is None and V is None)}) == 2
        assert (x is None) or (m is None and V is None)  #

        h = torch.cat((u, s), dim=-1) if u is not None else s
        switch_to_switch_dist = self.conditional_dist_transform(h)
        if self.F is not None:
            if m is not None:  # marginalise
                mp, Vp = filter_forward_prediction_step(
                    m=m,
                    V=V,
                    A=self.F,
                    R=switch_to_switch_dist.covariance_matrix,
                    b=switch_to_switch_dist.loc,
                )
            else:  # single sample fwd
                mp = matvec(self.F, x) + switch_to_switch_dist.loc
                Vp = switch_to_switch_dist.covariance_matrix
            return MultivariateNormal(loc=mp, scale_tril=cholesky(Vp))
        else:
            return switch_to_switch_dist
