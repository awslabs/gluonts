from typing import Sequence, Optional, Tuple
import torch
from inference.analytical_gausian_linear.inference_sequence_inhomogenous import (
    filter_forward,
    smooth_forward_backward,
    smooth_global,
    sample,
    loss_em,
    loss_forward,
)
from models_new_will_replace.base_gls import BaseGaussianLinearSystem, \
    ControlInputs
from models_new_will_replace.gls_homogenous import (
    ornstein_uhlenbeck_initialization,
)
from models_new_will_replace.base_gls import BaseGaussianLinearSystem, \
    Prediction, Latents, GLSVariables

from utils.utils import TensorDims


class GaussianLinearSystemInhomogenous(BaseGaussianLinearSystem):
    """
    GLS with inhomogenous (i.e. time-dependent) dynamics.

    Contains optionally a batch and particle dimension for the SSM parameters
    (default None).
    The purupose of this class is mainly for testing and sanity check,
    no practical purposes.
    """

    def __init__(
        self,
        n_state,
        n_target,
        n_ctrl_state,
        n_ctrl_target,
        n_timesteps,
        n_batch=None,
        n_particle=None,
        initialization_fn=ornstein_uhlenbeck_initialization,
    ):
        super().__init__(
            n_state=n_state,
            n_target=n_target,
            n_ctrl_state=n_ctrl_state,
            n_ctrl_target=n_ctrl_target,
        )
        self.n_timesteps = n_timesteps
        self.n_batch = n_batch
        self.n_particle = n_particle

        params = initialization_fn(
            n_obs=self.n_target,
            n_state=self.n_state,
            n_ctrl_state=self.n_ctrl_state,
            n_ctrl_obs=self.n_ctrl_target,
        )

        # 1x initial prior parameters
        self.m0 = torch.nn.Parameter(params.m0)
        self.LV0inv_tril = torch.nn.Parameter(params.LV0inv_tril)
        self.LV0inv_logdiag = torch.nn.Parameter(params.LV0inv_logdiag)

        # T-1 transition parameters
        dims_batch = (
            tuple([n_batch])
            if n_batch is not None and n_batch > 0
            else tuple()
        )
        dims_particle = (
            tuple([n_particle])
            if n_particle is not None and n_particle > 0
            else tuple()
        )
        dynamics_dims = (self.n_timesteps - 1,) + dims_particle + dims_batch
        measurement_dims = (self.n_timesteps,) + dims_particle + dims_batch

        self.A = torch.nn.Parameter(
            params.A.repeat(dynamics_dims + (1,) * params.A.ndim)
        )
        self.B = torch.nn.Parameter(
            params.B.repeat(dynamics_dims + (1,) * params.B.ndim)
        )
        self.LRinv_tril = torch.nn.Parameter(
            params.LRinv_tril.repeat(
                dynamics_dims + (1,) * params.LRinv_tril.ndim
            )
        )
        self.LRinv_logdiag = torch.nn.Parameter(
            params.LRinv_logdiag.repeat(
                dynamics_dims + (1,) * params.LRinv_logdiag.ndim
            )
        )

        # T measurement parameters
        self.C = torch.nn.Parameter(
            params.C.repeat(measurement_dims + (1,) * params.C.ndim)
        )
        self.D = torch.nn.Parameter(
            params.D.repeat(measurement_dims + (1,) * params.D.ndim)
        )
        self.LQinv_tril = torch.nn.Parameter(
            params.LQinv_tril.repeat(
                measurement_dims + (1,) * params.LQinv_tril.ndim
            )
        )
        self.LQinv_logdiag = torch.nn.Parameter(
            params.LQinv_logdiag.repeat(
                measurement_dims + (1,) * params.LQinv_logdiag.ndim
            )
        )

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
        loss = loss_forward(
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
        if dims.particle is not None and dims.particle > 0:
            loss = loss.mean(dim=0)
        # if dims.batch is not None and dims.batch > 0:
        #     loss = loss.sum(dim=0)
        return loss

    def _loss_em(
        self,
        past_targets: torch.Tensor,
        past_controls: Optional[ControlInputs] = None,
    ) -> torch.Tensor:
        dims = self.get_dims(y=past_targets)
        loss = loss_em(
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
        if dims.particle is not None and dims.particle > 0:
            loss = loss.mean(dim=0)
        # if dims.batch is not None and dims.batch > 0:
        #     loss = loss.sum(dim=0)
        return loss
