import math
from box import Box
import torch
from utils.utils import convert_ssm_params_to_model_params
from inference.analytical_gausian_linear.inference_sequence_homogenous import \
    filter_forward, smooth_forward_backward, smooth_global, sample, loss_em, \
    loss_forward
from models.dynamical_system import DynamicalSystem


def ornstein_uhlenbeck_initialization(n_obs, n_state, n_ctrl_state, n_ctrl_obs,
                                      n_theta=10, mu=0.0, sig=2.0, y=None):
    """ initialise to a default model with OU prior """
    if y is not None:
        mu_y = torch.mean(y, dim=1)
        sig_y = torch.std(y, dim=1)
    else:
        mu_y = torch.zeros(n_obs)
        sig_y = torch.ones(n_obs)

    # after n_theta steps decay to exp(-1)

    theta = 1.0 - math.exp(-1.0 / n_theta)
    m0 = mu * torch.ones((n_state,))
    V0 = (sig ** 2) * torch.eye(n_state)

    C = 1.0 * (1 / math.sqrt(n_state)) * torch.ones((n_obs, n_state))
    D = (1.0 / math.sqrt(n_ctrl_obs)) * torch.ones((n_obs, n_ctrl_obs))

    # sig_y is std of the marginal y. Q is var of likelihood for one instance
    Q = 0.1 * ((sig_y) ** 2) * torch.eye(n_obs)

    A = (1 - theta) * torch.eye(n_state)
    # A = torch.eye(n_state)
    B = (1.0 / math.sqrt(n_ctrl_state)) * torch.ones((n_state, n_ctrl_state))

    # choose such that AVAT + R = sig, where A is diag.
    R = (sig ** 2 - (1 - theta) ** 2) * torch.eye(n_state)

    ssm_params = Box(A=A, B=B, C=C, D=D, R=R, Q=Q, m0=m0, V0=V0)
    return convert_ssm_params_to_model_params(**ssm_params)


class GaussianLinearSystemHomogenous(DynamicalSystem):
    """
    Standard GLS with fixed set of parameters that do not change over time.
    This class has methods to perform inference (filter and smooth)
    and estimate the (negative) model evidence (loss_*).

    Note: implementations of the various inference methods are not
    computationally optimal in the sense that static parts could be precomputed.
    Instead, this class aims to be consistent with other implementations with
    inhomogenous dynamics or linear(ized) approximations of non-linear dynamics.
    """

    def __init__(self, n_state, n_obs, n_ctrl_state, n_ctrl_obs,
                 initialization_fn=ornstein_uhlenbeck_initialization):
        super().__init__(n_state=n_state, n_obs=n_obs,
                         n_ctrl_state=n_ctrl_state, n_ctrl_obs=n_ctrl_obs)

        params = initialization_fn(
            n_obs=self.n_obs,
            n_state=self.n_state,
            n_ctrl_state=self.n_ctrl_state,
            n_ctrl_obs=self.n_ctrl_obs,
        )
        self.A = torch.nn.Parameter(params.A)
        self.B = torch.nn.Parameter(params.B)
        self.C = torch.nn.Parameter(params.C)
        self.D = torch.nn.Parameter(params.D)
        self.m0 = torch.nn.Parameter(params.m0)
        self.LV0inv_tril = torch.nn.Parameter(params.LV0inv_tril)
        self.LV0inv_logdiag = torch.nn.Parameter(params.LV0inv_logdiag)
        self.LRinv_tril = torch.nn.Parameter(params.LRinv_tril)
        self.LRinv_logdiag = torch.nn.Parameter(params.LRinv_logdiag)
        self.LQinv_tril = torch.nn.Parameter(params.LQinv_tril)
        self.LQinv_logdiag = torch.nn.Parameter(params.LQinv_logdiag)

    def filter_forward(self, y, u_state=None, u_obs=None):
        dims = self.get_dims(y)
        return filter_forward(
            dims=dims, A=self.A, B=self.B, C=self.C, D=self.D,
            LQinv_tril=self.LQinv_tril, LQinv_logdiag=self.LQinv_logdiag,
            LRinv_tril=self.LRinv_tril, LRinv_logdiag=self.LRinv_logdiag,
            LV0inv_tril=self.LV0inv_tril, LV0inv_logdiag=self.LV0inv_logdiag,
            m0=self.m0,
            y=y, u_state=u_state, u_obs=u_obs,
        )

    def smooth_forward_backward(self, y, u_state=None, u_obs=None):
        dims = self.get_dims(y)
        return smooth_forward_backward(
            dims=dims, A=self.A, B=self.B, C=self.C, D=self.D,
            LQinv_tril=self.LQinv_tril, LQinv_logdiag=self.LQinv_logdiag,
            LRinv_tril=self.LRinv_tril, LRinv_logdiag=self.LRinv_logdiag,
            LV0inv_tril=self.LV0inv_tril, LV0inv_logdiag=self.LV0inv_logdiag,
            m0=self.m0,
            y=y, u_state=u_state, u_obs=u_obs,
        )

    def smooth_global(self, y, u_state=None, u_obs=None):
        dims = self.get_dims(y)
        return smooth_global(
            dims=dims, A=self.A, B=self.B, C=self.C, D=self.D,
            LQinv_tril=self.LQinv_tril, LQinv_logdiag=self.LQinv_logdiag,
            LRinv_tril=self.LRinv_tril, LRinv_logdiag=self.LRinv_logdiag,
            LV0inv_tril=self.LV0inv_tril, LV0inv_logdiag=self.LV0inv_logdiag,
            m0=self.m0,
            y=y, u_state=u_state, u_obs=u_obs,
        )

    def sample(self, u_state=None, u_obs=None, n_timesteps=None, n_batch=None):
        # TODO: make sample function take initial sample as input.
        #  and this method take either (m, V) or sample, or None (-> use prior).
        dims = self.get_dims(u_state=u_state, u_obs=u_obs,
                             n_timesteps=n_timesteps, n_batch=n_batch)
        return sample(
            dims=dims, A=self.A, B=self.B, C=self.C, D=self.D,
            LQinv_tril=self.LQinv_tril, LQinv_logdiag=self.LQinv_logdiag,
            LRinv_tril=self.LRinv_tril, LRinv_logdiag=self.LRinv_logdiag,
            LV0inv_tril=self.LV0inv_tril, LV0inv_logdiag=self.LV0inv_logdiag,
            m0=self.m0, u_state=u_state, u_obs=u_obs,
        )

    def loss_forward(self, y, u_state=None, u_obs=None):
        dims = self.get_dims(y)
        return loss_forward(
            dims=dims, A=self.A, B=self.B, C=self.C, D=self.D,
            LQinv_tril=self.LQinv_tril, LQinv_logdiag=self.LQinv_logdiag,
            LRinv_tril=self.LRinv_tril, LRinv_logdiag=self.LRinv_logdiag,
            LV0inv_tril=self.LV0inv_tril, LV0inv_logdiag=self.LV0inv_logdiag,
            m0=self.m0,
            y=y, u_state=u_state, u_obs=u_obs,
        )

    def loss_em(self, y, u_state=None, u_obs=None):
        dims = self.get_dims(y)
        return loss_em(
            dims=dims, A=self.A, B=self.B, C=self.C, D=self.D,
            LQinv_tril=self.LQinv_tril, LQinv_logdiag=self.LQinv_logdiag,
            LRinv_tril=self.LRinv_tril, LRinv_logdiag=self.LRinv_logdiag,
            LV0inv_tril=self.LV0inv_tril, LV0inv_logdiag=self.LV0inv_logdiag,
            m0=self.m0,
            y=y, u_state=u_state, u_obs=u_obs,
        )
