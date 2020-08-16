from typing import Sequence, Optional, Tuple
import math
from box import Box
import torch
from utils.utils import convert_ssm_params_to_model_params
from inference.analytical_gausian_linear.inference_sequence_homogenous import (
    filter_forward,
    smooth_forward_backward,
    smooth_global,
    sample,
    loss_em,
    loss_forward,
)
from models.base_gls import BaseGaussianLinearSystem, \
    Prediction, Latents, GLSVariables, ControlInputs

from utils.utils import TensorDims


def ornstein_uhlenbeck_initialization(
    n_obs,
    n_state,
    n_ctrl_state,
    n_ctrl_obs,
    n_theta=10,
    mu=0.0,
    sig=2.0,
    y=None,
):
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


class GaussianLinearSystemHomogenous(BaseGaussianLinearSystem):
    """
    Standard GLS with fixed set of parameters that do not change over time.
    This class has methods to perform inference (filter and smooth)
    and estimate the (negative) model evidence (loss_*).

    Note: implementations of the various inference methods are not
    computationally optimal in the sense that static parts could be precomputed.
    Instead, this class aims to be consistent with other implementations with
    inhomogenous dynamics or linear(ized) approximations of non-linear dynamics.
    """

    def __init__(
        self,
        n_state,
        n_target,
        n_ctrl_state,
        n_ctrl_target,
        initialization_fn=ornstein_uhlenbeck_initialization,
    ):
        super().__init__(
            n_state=n_state,
            n_target=n_target,
            n_ctrl_state=n_ctrl_state,
            n_ctrl_target=n_ctrl_target,
        )

        params = initialization_fn(
            n_obs=self.n_target,
            n_state=self.n_state,
            n_ctrl_state=self.n_ctrl_state,
            n_ctrl_obs=self.n_ctrl_target,
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

    def get_dims(
        self, y=None, u_state=None, u_obs=None, n_timesteps=None, n_batch=None
    ):
        if y is not None:
            n_timesteps = y.shape[0]
            n_batch = y.shape[1]
        elif u_state is not None:
            n_timesteps = u_state.shape[0]
            n_batch = u_state.shape[1]
        elif u_obs is not None:
            n_timesteps = u_obs.shape[0]
            n_batch = u_obs.shape[1]
        else:
            if n_timesteps is None and n_batch is None:
                raise Exception(
                    "either provide n_timesteps and n_batch directly, "
                    "or provide any of (y, u_state, u_obs, u_switch). "
                    f"Got following types: "
                    f"y: {type(y)}, "
                    f"u_state: {type(u_state)}, "
                    f"u_obs: {type(u_obs)}, "
                    f"n_timesteps: {type(n_timesteps)}, "
                    f"n_batch: {type(n_batch)}"
                )
        return TensorDims(
            timesteps=n_timesteps,
            particle=self.n_particle,
            batch=n_batch,
            state=self.n_state,
            target=self.n_target,
            ctrl_target=self.n_ctrl_target,
            ctrl_state=self.n_ctrl_state,
        )

    def filter(
        self,
        past_targets: torch.Tensor,
        past_controls: Optional[ControlInputs] = None,
    ) -> Sequence[Latents]:
        dims = self.get_dims(y=past_targets)
        m_filter, V_filter = filter_forward(
            dims=dims,
            A=self.A,
            B=self.B,
            C=self.C,
            D=self.D,
            LQinv_tril=self.LQinv_tril,
            LQinv_logdiag=self.LQinv_logdiag,
            LRinv_tril=self.LRinv_tril,
            LRinv_logdiag=self.LRinv_logdiag,
            LV0inv_tril=self.LV0inv_tril,
            LV0inv_logdiag=self.LV0inv_logdiag,
            m0=self.m0,
            y=past_targets,
            u_state=past_controls.state,
            u_obs=past_controls.target,
        )
        return [
            Latents(variables=GLSVariables(m=m, V=V, x=None))
            for m, V in zip(m_filter, V_filter)
        ]

    def smooth(
        self,
        past_targets: torch.Tensor,
        past_controls: Optional[ControlInputs] = None,
    ) -> Sequence[Latents]:

        m_smooth, V_smooth, Cov_smooth = self._smooth_forward_backward(
            past_targets=past_targets,
            past_controls=past_controls
        )
        return [
            Latents(variables=GLSVariables(m=m, V=V, x=None))
            for m, V in zip(m_smooth, V_smooth)  # covariances ignored here.
        ]

    def sample_generative(
        self,
        n_steps_forecast: int,
        n_batch: int,
        future_controls: Optional[ControlInputs],
        deterministic: bool,
        *args,
        **kwargs,
    ) -> Sequence[Prediction]:

        dims = self.get_dims(
            u_state=future_controls.state,
            u_obs=future_controls.target,
            n_timesteps=n_steps_forecast,
            n_batch=n_batch,
        )

        state_vars, emissions = sample(
            dims=dims,
            A=self.A,
            B=self.B,
            C=self.C,
            D=self.D,
            LQinv_tril=self.LQinv_tril,
            LQinv_logdiag=self.LQinv_logdiag,
            LRinv_tril=self.LRinv_tril,
            LRinv_logdiag=self.LRinv_logdiag,
            LV0inv_tril=self.LV0inv_tril,
            LV0inv_logdiag=self.LV0inv_logdiag,
            m0=self.m0,
            u_state=future_controls.state,
            u_obs=future_controls.target,
        )
        return [
            Prediction(
                latents=Latents(variables=GLSVariables(m=None, V=None, x=x)),
                emissions=y,
            )
            for x, y in zip(state_vars, emissions)
        ]

    def forecast(
        self,
        n_steps_forecast: int,
        initial_latent: Latents,
        future_controls: Optional[ControlInputs],
        deterministic: bool,
    ) -> Sequence[Prediction]:
        raise NotImplementedError("TODO")

    def predict(
        self,
        n_steps_forecast: int,
        past_targets: torch.Tensor,
        past_controls: Optional[ControlInputs],
        future_controls: Optional[ControlInputs],
        deterministic: bool,
    ) -> Sequence[Prediction]:
        raise NotImplementedError("TODO")

    def loss(
        self,
        past_targets: torch.Tensor,
        past_controls: Optional[ControlInputs] = None,
    ) -> torch.Tensor:
        return self._loss_em(
            past_targets=past_targets,
            past_controls=past_controls,
        )

    def _smooth_forward_backward(
        self,
        past_targets: torch.Tensor,
        past_controls: Optional[ControlInputs] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dims = self.get_dims(y=past_targets)
        m_smooth, V_smooth, Cov_smooth = smooth_forward_backward(
            dims=dims,
            A=self.A,
            B=self.B,
            C=self.C,
            D=self.D,
            LQinv_tril=self.LQinv_tril,
            LQinv_logdiag=self.LQinv_logdiag,
            LRinv_tril=self.LRinv_tril,
            LRinv_logdiag=self.LRinv_logdiag,
            LV0inv_tril=self.LV0inv_tril,
            LV0inv_logdiag=self.LV0inv_logdiag,
            m0=self.m0,
            y=past_targets,
            u_state=past_controls.state,
            u_obs=past_controls.target,
        )
        return m_smooth, V_smooth, Cov_smooth

    def _smooth_global(
        self,
        past_targets: torch.Tensor,
        past_controls: Optional[ControlInputs] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dims = self.get_dims(y=past_targets)
        m_smooth, V_smooth, Cov_smooth = smooth_global(
            dims=dims,
            A=self.A,
            B=self.B,
            C=self.C,
            D=self.D,
            LQinv_tril=self.LQinv_tril,
            LQinv_logdiag=self.LQinv_logdiag,
            LRinv_tril=self.LRinv_tril,
            LRinv_logdiag=self.LRinv_logdiag,
            LV0inv_tril=self.LV0inv_tril,
            LV0inv_logdiag=self.LV0inv_logdiag,
            m0=self.m0,
            y=past_targets,
            u_state=past_controls.state,
            u_obs=past_controls.target,
        )
        return m_smooth, V_smooth, Cov_smooth

    def _loss_forward(
        self,
        past_targets: torch.Tensor,
        past_controls: Optional[ControlInputs] = None,
    ) -> torch.Tensor:
        dims = self.get_dims(y=past_targets)
        return loss_forward(
            dims=dims,
            A=self.A,
            B=self.B,
            C=self.C,
            D=self.D,
            LQinv_tril=self.LQinv_tril,
            LQinv_logdiag=self.LQinv_logdiag,
            LRinv_tril=self.LRinv_tril,
            LRinv_logdiag=self.LRinv_logdiag,
            LV0inv_tril=self.LV0inv_tril,
            LV0inv_logdiag=self.LV0inv_logdiag,
            m0=self.m0,
            y=past_targets,
            u_state=past_controls.state,
            u_obs=past_controls.target,
        )

    def _loss_em(
        self,
        past_targets: torch.Tensor,
        past_controls: Optional[ControlInputs] = None,
    ) -> torch.Tensor:
        dims = self.get_dims(y=past_targets)
        return loss_em(
            dims=dims,
            A=self.A,
            B=self.B,
            C=self.C,
            D=self.D,
            LQinv_tril=self.LQinv_tril,
            LQinv_logdiag=self.LQinv_logdiag,
            LRinv_tril=self.LRinv_tril,
            LRinv_logdiag=self.LRinv_logdiag,
            LV0inv_tril=self.LV0inv_tril,
            LV0inv_logdiag=self.LV0inv_logdiag,
            m0=self.m0,
            y=past_targets,
            u_state=past_controls.state,
            u_obs=past_controls.target,
        )
