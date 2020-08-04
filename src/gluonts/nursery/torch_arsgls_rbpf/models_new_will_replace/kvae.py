from typing import Sequence, Optional, Union, Tuple
from dataclasses import dataclass
from box import Box
import numpy as np
import torch
from torch import nn
from torch.distributions import MultivariateNormal
from utils.utils import (
    make_inv_tril_parametrization,
    make_inv_tril_parametrization_from_cholesky,
    TensorDims,
)
from torch_extensions.ops import matvec
from inference.analytical_gausian_linear.inference_sequence_inhomogenous import (
    filter_forward,
    smooth_forward_backward,
    loss_em,
)
from inference.analytical_gausian_linear.inference_step import (
    filter_forward_predictive_distribution,
    filter_forward_prediction_step,
    filter_forward_measurement_step,
    smooth_backward_step,
)
from models_new_will_replace.base_gls import (
    ControlInputs,
    Latents,
    Prediction,
    GLSVariables,
)
from models_new_will_replace.base_amortized_gls import (
    BaseAmortizedGaussianLinearSystem,
)
from models_new_will_replace.gls_parameters.gls_parameters import GLSParameters
from torch_extensions.distributions.parametrised_distribution import (
    ParametrisedMultivariateNormal,
)


@dataclass
class ControlInputsKVAE(ControlInputs):

    encoder: torch.Tensor


@dataclass
class GLSVariablesKVAE(GLSVariables):

    auxiliary: torch.Tensor
    rnn_state: [torch.Tensor, Sequence[Tuple[torch.Tensor, torch.Tensor]]]
    m_auxiliary_variational: [torch.Tensor, None]
    V_auxiliary_variational: [torch.Tensor, None]


@dataclass
class LatentsKVAE(Latents):

    variables: GLSVariablesKVAE

    def __post_init__(self):
        if hasattr(super(), "__post_init__"):
            super().__post_init__()

        assert isinstance(self.variables, GLSVariablesKVAE)


class KalmanVariationalAutoEncoder(BaseAmortizedGaussianLinearSystem):
    def __init__(
        self,
        *args,
        n_auxiliary: int,
        measurement_model: nn.Module,
        rnn_switch_model: nn.RNNCellBase,
        reconstruction_weight: float = 1.0,
        rao_blackwellized: bool = False,
        **kwargs,
    ):
        kwargs.update({"n_ctrl_target": None})
        super().__init__(*args, **kwargs)
        self.n_auxiliary = n_auxiliary
        self.measurement_model = measurement_model
        self.rnn_switch_model = rnn_switch_model
        self.reconstruction_weight = reconstruction_weight
        self.rao_blackwellized = rao_blackwellized
        self.z_initial = torch.nn.Parameter(torch.zeros(self.n_auxiliary))

    def filter_step(
        self,
        lats_tm1: (LatentsKVAE, None),
        tar_t: torch.Tensor,
        ctrl_t: ControlInputs,
        tar_is_obs_t: Optional[torch.Tensor] = None,
    ) -> LatentsKVAE:
        is_initial_step = lats_tm1 is None
        if tar_is_obs_t is None:
            tar_is_obs_t = torch.ones(
                tar_t.shape[:-1], dtype=tar_t.dtype, device=tar_t.device,
            )

        # 1) Initial step must prepare previous latents with prior and learnt z.
        if is_initial_step:
            n_particle, n_batch = self.n_particle, len(tar_t)
            state_prior = self.state_prior_model(
                None, batch_shape_to_prepend=(n_particle, n_batch),
            )
            z_init = self.z_initial[None, None].repeat(n_particle, n_batch, 1)
            lats_tm1 = LatentsKVAE(
                variables=GLSVariablesKVAE(
                    m=state_prior.loc,
                    V=state_prior.covariance_matrix,
                    Cov=None,
                    x=None,
                    auxiliary=z_init,
                    rnn_state=None,
                    m_auxiliary_variational=None,
                    V_auxiliary_variational=None,
                ),
                gls_params=None,
            )
        # 2) Compute GLS params
        rnn_state_t, rnn_output_t = self.compute_deterministic_switch_step(
            rnn_input=lats_tm1.variables.auxiliary,
            rnn_prev_state=lats_tm1.variables.rnn_state,
        )
        gls_params_t = self.gls_base_parameters(
            switch=rnn_output_t, controls=ctrl_t,
        )

        # Perform filter step:
        # 3) Prediction Step: Only for t > 0 and using previous GLS params.
        # (In KVAE, they do first update then prediction step.)
        if is_initial_step:
            mp, Vp, = lats_tm1.variables.m, lats_tm1.variables.V
        else:
            mp, Vp = filter_forward_prediction_step(
                m=lats_tm1.variables.m,
                V=lats_tm1.variables.V,
                R=lats_tm1.gls_params.R,
                A=lats_tm1.gls_params.A,
                b=lats_tm1.gls_params.b,
            )
        # 4) Update step
        # 4a) Observed data: Infer pseudo-obs by encoding obs && Bayes update
        auxiliary_variational_dist_t = self.encoder(tar_t)
        z_infer_t = auxiliary_variational_dist_t.rsample([self.n_particle])
        m_infer_t, V_infer_t = filter_forward_measurement_step(
            y=z_infer_t,
            m=mp,
            V=Vp,
            Q=gls_params_t.Q,
            C=gls_params_t.C,
            d=gls_params_t.d,
        )

        # 4b) Choice: inferred / predicted m, V for observed / missing data.
        is_filtered = tar_is_obs_t[None, :].repeat(self.n_particle, 1).byte()
        replace_m_fw = is_filtered[:, :, None].repeat(1, 1, mp.shape[2])
        replace_V_fw = is_filtered[:, :, None, None].repeat(
            1, 1, Vp.shape[2], Vp.shape[3],
        )
        assert replace_m_fw.shape == m_infer_t.shape == mp.shape
        assert replace_V_fw.shape == V_infer_t.shape == Vp.shape

        m_t = torch.where(replace_m_fw, m_infer_t, mp)
        V_t = torch.where(replace_V_fw, V_infer_t, Vp)

        # 4c) Missing Data: Predict pseudo-observations && No Bayes update
        mpz_t, Vpz_t = filter_forward_predictive_distribution(
            m=m_t,  # posterior predictive or one-step-predictive (if missing)
            V=V_t,
            Q=gls_params_t.Q,
            C=gls_params_t.C,
            d=gls_params_t.d,
        )
        auxiliary_predictive_dist_t = MultivariateNormal(
            loc=mpz_t, covariance_matrix=Vpz_t,
        )
        z_gen_t = auxiliary_predictive_dist_t.rsample()

        # 4d) Choice: inferred / predicted z for observed / missing data.
        # One-step predictive if missing and inferred from encoder otherwise.
        replace_z = is_filtered[:, :, None].repeat(1, 1, z_gen_t.shape[2])
        z_t = torch.where(replace_z, z_infer_t, z_gen_t)

        # 5) Put result in Latents object, used in next iteration
        lats_t = LatentsKVAE(
            variables=GLSVariablesKVAE(
                m=m_t,
                V=V_t,
                Cov=None,
                x=None,
                auxiliary=z_t,
                rnn_state=rnn_state_t,
                m_auxiliary_variational=auxiliary_variational_dist_t.loc,
                V_auxiliary_variational=auxiliary_variational_dist_t.covariance_matrix,
            ),
            gls_params=gls_params_t,
        )
        return lats_t

    def smooth_step(
        self,
        lats_smooth_tp1: (LatentsKVAE, None),
        lats_filter_t: (LatentsKVAE, None),
    ) -> LatentsKVAE:
        # use the functional implementation given fixed params and filter dist.
        m_sm_t, V_sm_t, Cov_sm_t = smooth_backward_step(
            # <-- future smoothing part
            m_sm=lats_smooth_tp1.variables.m,
            V_sm=lats_smooth_tp1.variables.V,
            # --> past / current filter parts
            m_fw=lats_filter_t.variables.m,
            V_fw=lats_filter_t.variables.V,
            # --> forward-generated GLS params (there is no bw for them).
            A=lats_filter_t.gls_params.A,
            R=lats_filter_t.gls_params.R,
            b=lats_filter_t.gls_params.b,
        )

        # pack into into Latents object.
        lats_t = LatentsKVAE(
            variables=GLSVariablesKVAE(
                m=m_sm_t,
                V=V_sm_t,
                Cov=Cov_sm_t,
                x=None,
                auxiliary=lats_filter_t.variables.auxiliary,  # from forward
                rnn_state=lats_filter_t.variables.rnn_state,  # from forward
                m_auxiliary_variational=lats_filter_t.variables.m_auxiliary_variational,
                V_auxiliary_variational=lats_filter_t.variables.V_auxiliary_variational,
            ),
            gls_params=lats_filter_t.gls_params,  # from forward
        )
        return lats_t

    def sample_step(
        self,
        lats_tm1: LatentsKVAE,
        ctrl_t: ControlInputsKVAE,
        deterministic: bool = False,
    ) -> Prediction:
        first_step = lats_tm1.gls_params is None

        if first_step:  # from t == 0, i.e. lats_tm1 is t == -1.
            n_batch = len(lats_tm1.variables.auxiliary.shape[1])
            assert lats_tm1.variables.x is None
            assert lats_tm1.variables.m is None
            assert lats_tm1.variables.V is None
            x_t_dist = self.state_prior_model(
                None, batch_shape_to_prepend=(self.n_particle, n_batch)
            )
        else:
            x_t_dist = torch.distributions.MultivariateNormal(
                loc=(
                    matvec(lats_tm1.gls_params.A, lats_tm1.variables.x)
                    if lats_tm1.gls_params.A is not None
                    else lats_tm1.variables.x
                )
                + (
                    lats_tm1.gls_params.b
                    if lats_tm1.gls_params.b is not None
                    else 0.0
                ),
                scale_tril=lats_tm1.gls_params.LR,
            )

        rnn_state_t, rnn_output_t = self.compute_deterministic_switch_step(
            rnn_input=lats_tm1.variables.auxiliary,
            rnn_prev_state=lats_tm1.variables.rnn_state,
        )
        gls_params_t = self.gls_base_parameters(
            switch=rnn_output_t, controls=ctrl_t,
        )

        x_t = x_t_dist.mean if deterministic else x_t_dist.rsample()
        z_t_dist = torch.distributions.MultivariateNormal(
            loc=matvec(gls_params_t.C, x_t)
            + (gls_params_t.d if gls_params_t.d is not None else 0.0),
            covariance_matrix=gls_params_t.Q,
        )
        z_t = z_t_dist.mean if deterministic else z_t_dist.rsample()

        lats_t = LatentsKVAE(
            variables=GLSVariablesKVAE(
                m=None,
                V=None,
                Cov=None,
                x=x_t,
                auxiliary=z_t,
                rnn_state=rnn_state_t,
                m_auxiliary_variational=None,
                V_auxiliary_variational=None,
            ),
            gls_params=gls_params_t,
        )

        emission_dist_t = self.emit(lats_t=lats_t, ctrl_t=ctrl_t)
        emissions_t = emission_dist_t.mean \
            if deterministic \
            else emission_dist_t.sample()

        return Prediction(
            latents=lats_t,
            emissions=emissions_t,
        )

    def loss(
        self,
        past_targets: [Sequence[torch.Tensor], torch.Tensor],
        past_controls: Optional[
            Union[Sequence[ControlInputs], ControlInputs]
        ] = None,
        past_targets_is_observed: Optional[
                Union[Sequence[torch.Tensor], torch.Tensor]] = None,
    ) -> torch.Tensor:

        if self.rao_blackwellized:
            if past_targets_is_observed is None:
                return self._loss_em_rb_efficient(
                    past_targets=past_targets,
                    past_controls=past_controls,
                )
            else:
                raise NotImplementedError(
                    "did not yet implement Rao-BW loss with missing data"
                )
        else:
            if past_targets_is_observed is None:
                return self._loss_em_mc_efficient(
                    past_targets=past_targets,
                    past_controls=past_controls,
                )
            else:
                return self._loss_em_mc(
                    past_targets=past_targets,
                    past_controls=past_controls,
                    past_targets_is_observed=past_targets_is_observed,
                )

    def _loss_em_mc(
        self,
        past_targets: [Sequence[torch.Tensor], torch.Tensor],
        past_controls: Optional[
            Union[Sequence[ControlInputs], ControlInputs]
        ] = None,
        past_targets_is_observed: Optional[
            Union[Sequence[torch.Tensor], torch.Tensor]] = None,
    ) -> torch.Tensor:
        """" Monte Carlo loss as computed in KVAE paper """
        n_batch = len(past_targets[0])

        past_controls = self._expand_particle_dim(past_controls)

        # A) SSM related distributions:
        # A1) smoothing.
        latents_smoothed = self.smooth(
            past_targets=past_targets,
            past_controls=past_controls,
            past_targets_is_observed=past_targets_is_observed,
        )
        m = torch.stack([l.variables.m for l in latents_smoothed])
        V = torch.stack([l.variables.V for l in latents_smoothed])
        z = torch.stack([l.variables.auxiliary for l in latents_smoothed])
        state_smoothed_dist = MultivariateNormal(loc=m, covariance_matrix=V)
        x = state_smoothed_dist.rsample()

        A = torch.stack([l.gls_params.A for l in latents_smoothed])
        C = torch.stack([l.gls_params.C for l in latents_smoothed])
        LR = torch.stack([l.gls_params.LR for l in latents_smoothed])
        LQ = torch.stack([l.gls_params.LQ for l in latents_smoothed])
        if latents_smoothed[0].gls_params.B is not None:
            B = torch.stack([l.gls_params.B for l in latents_smoothed])
        else:
            B = None
        if latents_smoothed[0].gls_params.D is not None:
            D = torch.stack([l.gls_params.D for l in latents_smoothed])
        else:
            D = None

        # A2) prior && posterior transition distribution.
        prior_dist = self.state_prior_model(
            None, batch_shape_to_prepend=(self.n_particle, n_batch)
        )

        #  # A, B, R are already 0:T-1.
        transition_dist = MultivariateNormal(
            loc=matvec(A[:-1], x[:-1])
            + (
                matvec(B[:-1], past_controls.state[:-1])
                if B is not None
                else 0.0
            ),
            scale_tril=LR[:-1],
        )
        # A3) posterior predictive (auxiliary) distribution.
        auxiliary_predictive_dist = MultivariateNormal(
            loc=matvec(C, x)
            + (matvec(D, past_controls.target) if D is not None else 0.0),
            scale_tril=LQ,
        )

        # A4) SSM related losses
        # mean over particle dim, sum over time (after masking), leave batch dim
        l_prior = -prior_dist.log_prob(x[0:1]).mean(dim=1).sum(dim=0)
        l_transition = -transition_dist.log_prob(x[1:]).mean(dim=1).sum(dim=0)
        l_entropy = state_smoothed_dist.log_prob(x).mean(dim=1).sum(dim=0)

        _l_aux_timewise = -auxiliary_predictive_dist.log_prob(z).mean(dim=1)
        if past_targets_is_observed is not None:
            _l_aux_timewise = _l_aux_timewise * past_targets_is_observed
        l_auxiliary = _l_aux_timewise.sum(dim=0)

        # B) VAE related distributions
        # B1) inv_measurement_dist already obtained from smoothing (as we dont want to re-compute)
        # B2) measurement (decoder) distribution
        # transpose TPBF -> PTBF to broadcast log_prob of y (TBF) correctly
        z_particle_first = z.transpose(0, 1)
        measurement_dist = self.measurement_model(z_particle_first)
        # B3) VAE related losses
        # We use z_particle_first for correct broadcasting -> dim=0 is particle.
        _l_meas_timewise = -measurement_dist.log_prob(past_targets).mean(dim=0)
        if past_targets_is_observed is not None:
            _l_meas_timewise = _l_meas_timewise * past_targets_is_observed
        l_measurement = _l_meas_timewise.sum(dim=0)

        auxiliary_variational_dist = MultivariateNormal(
            loc=torch.stack(
                [l.variables.m_auxiliary_variational for l in latents_smoothed]
            ),
            covariance_matrix=torch.stack(
                [l.variables.V_auxiliary_variational for l in latents_smoothed]
            ),
        )
        _l_variational_timewise = auxiliary_variational_dist.log_prob(
            z_particle_first
        ).mean(dim=0)  # again dim=0 is particle dim here.
        if past_targets_is_observed is not None:
            _l_variational_timewise = _l_variational_timewise * past_targets_is_observed
        l_inv_measurement = _l_variational_timewise.sum(dim=0)

        assert all(
            t.shape == l_prior.shape
            for t in (
                l_prior,
                l_transition,
                l_auxiliary,
                l_measurement,
                l_inv_measurement,
            )
        )

        l_total = (
            self.reconstruction_weight * l_measurement
            + l_inv_measurement
            + l_auxiliary
            + l_prior
            + l_transition
            + l_entropy
        )
        return l_total

    def emit(
        self, lats_t: LatentsKVAE, ctrl_t: ControlInputs,
    ) -> torch.distributions.Distribution:
        return self.measurement_model(lats_t.variables.auxiliary)

    def compute_deterministic_switch_sequence(
        self, rnn_inputs: torch.Tensor,
    ) -> Tuple[Sequence[Union[Tuple, torch.Tensor]], torch.Tensor]:
        (T, P, B, F,) = rnn_inputs.shape
        rnn_inputs_flat = rnn_inputs.reshape([T, P * B, F])

        rnn_states = [None] * len(rnn_inputs)
        for t in range(len(rnn_inputs)):
            rnn_state_flat_t = self.rnn_switch_model(
                input=rnn_inputs_flat[t],
                hx=rnn_state_flat_t if t > 0 else None,
            )
            if isinstance(rnn_state_flat_t, Tuple):
                rnn_states[t] = tuple(
                    _h.reshape([P, B, _h.shape[-1]]) for _h in rnn_state_flat_t
                )
            else:
                rnn_states[t] = rnn_state_flat_t.reshape(
                    [P, B, rnn_state_flat_t.shape[-1]],
                )

        if isinstance(rnn_states[0], Tuple):
            rnn_outputs = torch.stack(
                [rnn_states[t][0] for t in range(T)], dim=0
            )
        else:
            rnn_outputs = torch.stack(rnn_states, dim=0)

        return rnn_states, rnn_outputs

    def compute_deterministic_switch_step(
        self,
        rnn_input: torch.Tensor,
        rnn_prev_state: Union[Tuple[torch.Tensor], torch.Tensor, None],
    ) -> Tuple[Union[Tuple, torch.Tensor], torch.Tensor]:
        (P, B, F,) = rnn_input.shape
        rnn_inputs_flat = rnn_input.reshape([P * B, F])
        if isinstance(rnn_prev_state, Tuple):
            hx_flat = tuple(
                _h.reshape([P * B, _h.shape[-1]]) for _h in rnn_prev_state
            )
        else:
            if rnn_prev_state is not None:
                hx_flat = rnn_prev_state.reshape(
                    [P * B, rnn_prev_state.shape[-1]]
                )
            else:
                hx_flat = None  # will be handled by the RNN.
        h_flat = self.rnn_switch_model(input=rnn_inputs_flat, hx=hx_flat,)
        if isinstance(h_flat, Tuple):
            rnn_state = tuple(
                _h.reshape([P, B, _h.shape[-1]]) for _h in h_flat
            )
        else:
            rnn_state = h_flat.reshape([P, B, h_flat.shape[-1]])

        if isinstance(rnn_state, Tuple):
            rnn_output = rnn_state[0]
        else:
            rnn_output = rnn_state
        return rnn_state, rnn_output

    def _prepare_forecast(
        self,
        initial_latent: Latents,
        controls: Optional[
            Union[Sequence[ControlInputs], ControlInputs]
        ] = None,
        deterministic: bool = False,
    ):
        return initial_latent, controls

    def _sample_initial_latents(self, n_particle, n_batch) -> LatentsKVAE:
        # only use rnn_state (None at t==-1) and initial auxiliary variable
        # to compute the initial GLS Params. Prior is already t == 0.
        z_initial = self.z_initial[None, None, :].repeat(
            self.n_particle, n_batch, 1,
        )
        initial_latents = LatentsKVAE(
            gls_params=None,
            variables=GLSVariablesKVAE(
                x=None,
                m=None,
                V=None,
                auxiliary=z_initial,
                rnn_state=None,
            )
        )
        return initial_latents

    def _expand_particle_dim(self, controls: ControlInputs):
        # assumes we have time dimension
        for key, val in controls.__dict__.items():
            if (val is not None) and ((val.ndim == 3) and (val.shape[1] != 1)):
                setattr(controls, key, val.unsqueeze(dim=1))
        return controls

    # Below are some more efficient method implementations that batch time-dim.
    def _loss_em_rb_efficient(
            self,
            past_targets: [Sequence[torch.Tensor], torch.Tensor],
            past_controls: Optional[
                Union[Sequence[ControlInputs], ControlInputs]
            ] = None,
    ) -> torch.Tensor:
        """
        Rao-Blackwellization for part of the loss (the EM loss term of the SSM).
        """
        n_batch = len(past_targets[0])
        past_controls = self._expand_particle_dim(past_controls)

        # auxiliary variables (z) are SSM pseudo observations.
        # For log_prob evaluation, need particle_first, for RNN time_first.
        q = self.encoder(past_targets)
        z_particle_first = q.rsample([self.n_particle])
        z_time_first = z_particle_first.transpose(0, 1)
        z_initial = self.z_initial[None, None, None, :].repeat(
            1, self.n_particle, n_batch, 1,
        )

        # Unroll RNN on all pseudo-obervations to get the SSM params
        rnn_states, rnn_outputs = self.compute_deterministic_switch_sequence(
            rnn_inputs=torch.cat([z_initial, z_time_first[:-1]], dim=0),
        )
        gls_params = self.gls_base_parameters(
            switch=rnn_outputs, controls=past_controls,
        )

        (
            LQinv_tril,
            LQinv_logdiag,
        ) = make_inv_tril_parametrization_from_cholesky(gls_params.LQ, )
        (
            LRinv_tril,
            LRinv_logdiag,
        ) = make_inv_tril_parametrization_from_cholesky(gls_params.LR, )

        state_prior = self.state_prior_model(
            None, batch_shape_to_prepend=(self.n_particle, n_batch),
        )
        LV0inv_tril, LV0inv_logdiag = make_inv_tril_parametrization(
            state_prior.covariance_matrix,
        )
        m0 = state_prior.loc

        dims = Box(
            timesteps=len(past_targets),
            target=self.n_auxiliary,
            state=self.n_state,
            particle=self.n_particle,
            batch=n_batch,
        )

        _l_em_particle_batch_wise = loss_em(
            dims=dims,
            # contain obs which is auxiliary here.
            A=gls_params.A[:-1],
            B=gls_params.B[:-1] if gls_params.B is not None else None,
            LRinv_tril=LRinv_tril[:-1],
            LRinv_logdiag=LRinv_logdiag[:-1],
            C=gls_params.C,
            D=gls_params.D,
            LQinv_tril=LQinv_tril,
            LQinv_logdiag=LQinv_logdiag,
            LV0inv_tril=LV0inv_tril,
            LV0inv_logdiag=LV0inv_logdiag,
            m0=m0,
            y=z_time_first,
            u_state=past_controls.state,
            u_obs=past_controls.target,
        )
        l_em = _l_em_particle_batch_wise.sum(dim=0) / dims.particle

        l_measurement = (
                -self.measurement_model(z_particle_first)
                .log_prob(past_targets)
                .sum(dim=(0, 1))
                / dims.particle
        )
        l_auxiliary_encoder = (
                q.log_prob(z_particle_first).sum(dim=(0, 1)) / dims.particle
        )

        assert all(
            l.shape == l_measurement.shape
            for l in (l_measurement, l_auxiliary_encoder, l_em)
        )
        l_total = (
                self.reconstruction_weight * l_measurement
                + l_auxiliary_encoder
                + l_em
        )
        return l_total

    def _loss_em_mc_efficient(
            self,
            past_targets: [Sequence[torch.Tensor], torch.Tensor],
            past_controls: Optional[
                Union[Sequence[ControlInputs], ControlInputs]
            ] = None,
    ) -> torch.Tensor:
        """
        Monte Carlo loss as computed in KVAE paper.
        Can be computed more efficiently if no missing data (no imputation),
        by batching some things along time-axis.
        """
        past_controls = self._expand_particle_dim(past_controls)
        n_batch = len(past_targets[0])

        # A) SSM related distributions:
        # A1) smoothing.
        latents_smoothed = self._smooth_efficient(
            past_targets=past_targets, past_controls=past_controls,
            return_time_tensor=True,
        )

        state_smoothed_dist = MultivariateNormal(
            loc=latents_smoothed.variables.m,
            covariance_matrix=latents_smoothed.variables.V,
        )
        x = state_smoothed_dist.rsample()
        gls_params = latents_smoothed.gls_params

        # A2) prior && posterior transition distribution.
        prior_dist = self.state_prior_model(
            None, batch_shape_to_prepend=(self.n_particle, n_batch)
        )

        #  # A, B, R are already 0:T-1.
        transition_dist = MultivariateNormal(
            loc=matvec(gls_params.A[:-1], x[:-1])
                + (
                    matvec(gls_params.B[:-1], past_controls.state[:-1])
                    if gls_params.B is not None
                    else 0.0
                ),
            covariance_matrix=gls_params.R[:-1],
        )
        # A3) posterior predictive (auxiliary) distribution.
        auxiliary_predictive_dist = MultivariateNormal(
            loc=matvec(gls_params.C, x)
                + (
                    matvec(gls_params.D, past_controls.target)
                    if gls_params.D is not None
                    else 0.0
                ),
            covariance_matrix=gls_params.Q,
        )

        # A4) SSM related losses
        l_prior = (
                -prior_dist.log_prob(x[0:1]).sum(dim=(0, 1)) / self.n_particle
        )  # time and particle dim
        l_transition = (
                -transition_dist.log_prob(x[1:]).sum(
                    dim=(0, 1)) / self.n_particle
        )  # time and particle dim
        l_auxiliary = (
                -auxiliary_predictive_dist.log_prob(
                    latents_smoothed.variables.auxiliary
                ).sum(dim=(0, 1))
                / self.n_particle
        )  # time and particle dim
        l_entropy = (
                state_smoothed_dist.log_prob(x).sum(
                    dim=(0, 1))  # negative entropy
                / self.n_particle
        )  # time and particle dim

        # B) VAE related distributions
        # B1) inv_measurement_dist already obtained from smoothing (as we dont want to re-compute)
        # B2) measurement (decoder) distribution
        # transpose TPBF -> PTBF to broadcast log_prob of y (TBF) correctly
        z_particle_first = latents_smoothed.variables.auxiliary.transpose(0, 1)
        measurement_dist = self.measurement_model(z_particle_first)
        # B3) VAE related losses
        l_measurement = (
                -measurement_dist.log_prob(past_targets).sum(dim=(0, 1))
                / self.n_particle
        )  # time and particle dim

        auxiliary_variational_dist = MultivariateNormal(
            loc=latents_smoothed.variables.m_auxiliary_variational,
            covariance_matrix=latents_smoothed.variables.V_auxiliary_variational,
        )
        l_inv_measurement = (
                auxiliary_variational_dist.log_prob(z_particle_first).sum(
                    dim=(0, 1)
                )
                / self.n_particle
        )  # time and particle dim

        assert all(
            t.shape == l_prior.shape
            for t in (
                l_prior,
                l_transition,
                l_auxiliary,
                l_measurement,
                l_inv_measurement,
            )
        )

        l_total = (
                self.reconstruction_weight * l_measurement
                + l_inv_measurement
                + l_auxiliary
                + l_prior
                + l_transition
                + l_entropy
        )
        return l_total

    def _filter_efficient(
        self,
        past_targets: [Sequence[torch.Tensor], torch.Tensor],
        past_controls: Optional[
            Union[Sequence[ControlInputs], ControlInputs]
        ] = None,
        return_time_tensor: bool = False,
    ) -> [Sequence[LatentsKVAE], LatentsKVAE]:
        """
        More efficient filter implementation when no data is missing.
        Smoothing that re-uses a functional implementation of general smoothing
        in a standard GLS.
        Computes auxiliary variables and efficiently by batching over time.
        Computes GLS parameters efficiently by first unrolling the whole RNN.
        """
        n_batch = len(past_targets[0])

        past_controls = self._expand_particle_dim(past_controls)

        state_prior = self.state_prior_model(
            None, batch_shape_to_prepend=(self.n_particle, n_batch),
        )
        LV0inv_tril, LV0inv_logdiag = make_inv_tril_parametrization(
            state_prior.covariance_matrix,
        )
        m0 = state_prior.loc

        # Encode observations y[0:T] to obtain all pseudo-observation z[0:T]
        auxiliary_variational_dist = self.encoder(past_targets)
        z = auxiliary_variational_dist.rsample([self.n_particle]).transpose(0, 1)
        # Use as RNN input [z_initial, z[0:T-1]], i.e. previous pseudo-observation.
        z_initial = self.z_initial[None, None, None, :].repeat(
            1, self.n_particle, n_batch, 1,
        )

        # Unroll RNN on all pseudo-obervations to get the SSM params
        rnn_states, rnn_outputs = self.compute_deterministic_switch_sequence(
            rnn_inputs=torch.cat([z_initial, z[:-1]], dim=0),
        )
        gls_params = self.gls_base_parameters(
            switch=rnn_outputs, controls=past_controls,
        )

        # filter with pseudo-obs.
        (
            LQinv_tril,
            LQinv_logdiag,
        ) = make_inv_tril_parametrization_from_cholesky(gls_params.LQ,)
        (
            LRinv_tril,
            LRinv_logdiag,
        ) = make_inv_tril_parametrization_from_cholesky(gls_params.LR,)

        # temporary replacement hack.
        dims = Box(
            timesteps=len(past_targets),
            obs=self.n_auxiliary,
            state=self.n_state,
            particle=self.n_particle,
            batch=n_batch,
        )
        m_fw, V_fw = filter_forward(
            dims=dims,
            # contain obs which is auxiliary here.
            A=gls_params.A[:-1],
            B=gls_params.B[:-1] if gls_params.B is not None else None,
            LRinv_tril=LRinv_tril[:-1],
            LRinv_logdiag=LRinv_logdiag[:-1],
            C=gls_params.C,
            D=gls_params.D,
            LQinv_tril=LQinv_tril,
            LQinv_logdiag=LQinv_logdiag,
            LV0inv_tril=LV0inv_tril,
            LV0inv_logdiag=LV0inv_logdiag,
            m0=m0,
            y=z,
            u_state=past_controls.state,
            u_obs=past_controls.target,
        )

        filtered_latents_with_time_dim = LatentsKVAE(
            variables=GLSVariablesKVAE(
                m=m_fw,
                V=V_fw,
                Cov=None,
                x=None,
                auxiliary=z,
                rnn_state=rnn_states,
                m_auxiliary_variational=auxiliary_variational_dist.loc,
                V_auxiliary_variational=auxiliary_variational_dist.covariance_matrix,
            ),
            gls_params=gls_params,
        )
        if return_time_tensor:
            return filtered_latents_with_time_dim
        else:
            return list(iter(filtered_latents_with_time_dim))

    def _smooth_efficient(
        self,
        past_targets: [Sequence[torch.Tensor], torch.Tensor],
        past_controls: Optional[
            Union[Sequence[ControlInputs], ControlInputs]
        ] = None,
        return_time_tensor: bool = False,
    ) -> [Sequence[LatentsKVAE], LatentsKVAE]:
        """
        More efficient smoothing implementation when no data is missing.
        Smoothing that re-uses a functional implementation of general smoothing
        in a standard GLS.
        Computes auxiliary variables and efficiently by batching over time.
        Computes GLS parameters efficiently by first unrolling the whole RNN.
        Optionally returns the Latents with a time-dimension,
        in order to compute the loss also more efficiently
        """
        n_batch = len(past_targets[0])

        past_controls = self._expand_particle_dim(past_controls)

        state_prior = self.state_prior_model(
            None, batch_shape_to_prepend=(self.n_particle, n_batch)
        )
        LV0inv_tril, LV0inv_logdiag = make_inv_tril_parametrization(
            state_prior.covariance_matrix
        )
        m0 = state_prior.loc

        auxiliary_variational_dist = self.encoder(past_targets)
        z = auxiliary_variational_dist.rsample([self.n_particle]).transpose(
            0, 1,
        )

        z_initial = self.z_initial[None, None, None, :].repeat(
            1, self.n_particle, n_batch, 1,
        )

        # Unroll RNN on all pseudo-obervations to get the SSM params
        rnn_states, rnn_outputs = self.compute_deterministic_switch_sequence(
            rnn_inputs=torch.cat([z_initial, z[:-1]], dim=0),
        )
        gls_params = self.gls_base_parameters(
            switch=rnn_outputs, controls=past_controls,
        )

        LQinv_tril, LQinv_logdiag = make_inv_tril_parametrization(gls_params.Q)
        LRinv_tril, LRinv_logdiag = make_inv_tril_parametrization(gls_params.R)

        dims = Box(
            timesteps=len(past_targets),
            target=self.n_auxiliary,
            state=self.n_state,
            particle=self.n_particle,
            batch=n_batch,
        )
        m_sm, V_sm, Cov_sm = smooth_forward_backward(
            dims=dims,
            # contain obs which is auxiliary here.
            A=gls_params.A[:-1],
            B=gls_params.B[:-1] if gls_params.B is not None else None,
            LRinv_tril=LRinv_tril[:-1],
            LRinv_logdiag=LRinv_logdiag[:-1],
            C=gls_params.C,
            D=gls_params.D,
            LQinv_tril=LQinv_tril,
            LQinv_logdiag=LQinv_logdiag,
            LV0inv_tril=LV0inv_tril,
            LV0inv_logdiag=LV0inv_logdiag,
            m0=m0,
            y=z,
            u_state=past_controls.state,
            u_obs=past_controls.target,
        )

        smoothed_latents_with_time_dim = LatentsKVAE(
            variables=GLSVariablesKVAE(
                m=m_sm,
                V=V_sm,
                Cov=None,
                x=None,
                auxiliary=z,
                rnn_state=rnn_states,
                m_auxiliary_variational=auxiliary_variational_dist.loc,
                V_auxiliary_variational=auxiliary_variational_dist.covariance_matrix,
            ),
            gls_params=gls_params,
        )

        if return_time_tensor:
            return smoothed_latents_with_time_dim
        else:
            return list(iter(smoothed_latents_with_time_dim))

