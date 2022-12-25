from typing import Optional, Sequence, Union, Tuple
from dataclasses import dataclass

from box import Box

import torch
from torch import nn
from torch.distributions import MultivariateNormal

from models.base_amortized_gls import (
    LatentsRBSMC,
    Prediction,
)
from models.base_rbpf_gls import BaseRBSMCGaussianLinearSystem
from models.base_gls import GLSVariables, ControlInputs

from inference.smc.resampling import (
    resample,
    EffectiveSampleSizeResampleCriterion,
    systematic_resampling_indices,
)
from inference.smc.normalize import normalize_log_weights
from inference.analytical_gausian_linear.inference_step import (
    filter_forward_prediction_step,
    filter_forward_measurement_step,
    filter_forward_predictive_distribution,
)
from torch_extensions.ops import (
    matvec,
    batch_diag_matrix,
    batch_diag,
    cholesky,
)
from torch_extensions.distributions.parametrised_distribution import (
    ParametrisedDistribution,
    ParametrisedMultivariateNormal,
    prepend_batch_dims,
)
from torch_extensions.distributions.conditional_parametrised_distribution import (
    ParametrisedConditionalDistribution,
)
from models.gls_parameters.gls_parameters import GLSParameters


@dataclass
class ControlInputsSGLS(ControlInputs):
    switch: torch.Tensor
    encoder: torch.Tensor


@dataclass
class ControlInputsSGLSISSM(ControlInputsSGLS):
    seasonal_indicators: torch.Tensor


@dataclass
class GLSVariablesSGLS(GLSVariables):

    switch: torch.Tensor


@dataclass
class LatentsSGLS(LatentsRBSMC):

    variables: GLSVariablesSGLS
    log_weights: torch.Tensor

    def __post_init__(self):
        assert isinstance(self.variables, GLSVariablesSGLS)


class SwitchingGaussianLinearSystemBaseRBSMC(BaseRBSMCGaussianLinearSystem):
    def __init__(
        self,
        n_state: int,
        n_target: int,
        n_ctrl_state: int,
        n_ctrl_target: int,
        n_particle: int,
        n_switch: int,
        gls_base_parameters: GLSParameters,
        encoder: ParametrisedConditionalDistribution,
        state_prior_model: ParametrisedMultivariateNormal,
        switch_prior_model: ParametrisedDistribution,
        switch_transition_model: nn.Module,
        resampling_criterion_fn=EffectiveSampleSizeResampleCriterion(
            min_ess_ratio=0.5
        ),
        resampling_indices_fn: callable = systematic_resampling_indices,
    ):
        super().__init__(
            n_state=n_state,
            n_target=n_target,
            n_ctrl_state=n_ctrl_state,
            n_ctrl_target=n_ctrl_target,
            n_particle=n_particle,
            gls_base_parameters=gls_base_parameters,
            encoder=encoder,
            state_prior_model=state_prior_model,
            resampling_criterion_fn=resampling_criterion_fn,
            resampling_indices_fn=resampling_indices_fn,
        )
        self.n_switch = n_switch
        self.switch_prior_model = switch_prior_model
        self.switch_transition_model = switch_transition_model

    def filter_step(
        self,
        lats_tm1: (LatentsSGLS, None),
        tar_t: torch.Tensor,
        ctrl_t: ControlInputsSGLS,
        tar_is_obs_t: Optional[torch.Tensor] = None,
    ):
        if tar_is_obs_t is not None:
            raise NotImplementedError("cannot handle missing data atm.")

        is_initial_step = lats_tm1 is None
        if is_initial_step:
            n_particle, n_batch = self.n_particle, len(tar_t)
            state_prior = self.state_prior_model(
                None, batch_shape_to_prepend=(n_particle, n_batch),
            )
            log_norm_weights = normalize_log_weights(
                log_weights=torch.zeros_like(state_prior.loc[..., 0]),
            )
            lats_tm1 = LatentsSGLS(
                log_weights=None,  # Not used. We use log_norm_weights instead.
                gls_params=None,  # First (previous) step no gls_params
                variables=GLSVariablesSGLS(
                    m=state_prior.loc,
                    V=state_prior.covariance_matrix,
                    Cov=None,
                    x=None,
                    switch=None,
                ),
            )
            switch_model_dist = self._make_switch_prior_dist(
                lat_vars_tm1=lats_tm1.variables,
                ctrl_t=ctrl_t,
                n_particle=n_particle,
                n_batch=n_batch,
            )
        else:
            log_norm_weights = normalize_log_weights(
                log_weights=lats_tm1.log_weights,
            )
            log_norm_weights, resampled_tensors = resample(
                n_particle=self.n_particle,
                log_norm_weights=log_norm_weights,
                tensors_to_resample={
                    key: val
                    for key, val in lats_tm1.variables.__dict__.items()
                    if key not in ("x", "Cov")  # below set to None explicitly
                },
                resampling_indices_fn=self.resampling_indices_fn,
                criterion_fn=self.resampling_criterion_fn,
            )
            # We dont use gls_params anymore.
            # If needed for e.g. evaluation, remember to re-sample all params!
            lats_tm1 = LatentsSGLS(
                log_weights=None,  # Not used. We use log_norm_weights instead.
                gls_params=None,  # not used outside this function. Read above.
                variables=GLSVariablesSGLS(
                    **resampled_tensors, x=None, Cov=None,
                ),
            )
            switch_model_dist = self._make_switch_transition_dist(
                lat_vars_tm1=lats_tm1.variables, ctrl_t=ctrl_t,
            )

        switch_proposal_dist = self._make_switch_proposal_dist(
            switch_model_dist=switch_model_dist,
            switch_encoder_dist=self._make_encoder_dists(
                tar_t=tar_t, ctrl_t=ctrl_t,
            ).switch,
        )
        s_t = switch_proposal_dist.rsample()
        gls_params_t = self.gls_base_parameters(switch=s_t, controls=ctrl_t,)

        mp, Vp = filter_forward_prediction_step(
            m=lats_tm1.variables.m,
            V=lats_tm1.variables.V,
            R=gls_params_t.R,
            A=gls_params_t.A,
            b=gls_params_t.b,
        )

        m_t, V_t = filter_forward_measurement_step(
            y=tar_t,
            m=mp,
            V=Vp,
            Q=gls_params_t.Q,
            C=gls_params_t.C,
            d=gls_params_t.d,
        )
        mpy_t, Vpy_t = filter_forward_predictive_distribution(
            m=mp, V=Vp, Q=gls_params_t.Q, C=gls_params_t.C, d=gls_params_t.d,
        )
        measurement_dist = MultivariateNormal(
            loc=mpy_t, scale_tril=cholesky(Vpy_t),
        )

        log_update = (
            measurement_dist.log_prob(tar_t)
            + switch_model_dist.log_prob(s_t)
            - switch_proposal_dist.log_prob(s_t)
        )
        log_weights_t = log_norm_weights + log_update

        return LatentsSGLS(
            log_weights=log_weights_t,
            gls_params=None,  # not used outside this function
            variables=GLSVariablesSGLS(
                m=m_t, V=V_t, x=None, Cov=None, switch=s_t,
            ),
        )

    def sample_step(
        self,
        lats_tm1: LatentsSGLS,
        ctrl_t: ControlInputsSGLS,
        deterministic: bool = False,
    ) -> Prediction:
        n_batch = lats_tm1.variables.x.shape[1]

        if lats_tm1.variables.switch is None:
            switch_model_dist_t = self._make_switch_prior_dist(
                n_particle=self.n_particle,
                n_batch=n_batch,
                lat_vars_tm1=lats_tm1.variables,
                ctrl_t=ctrl_t,
            )
        else:
            switch_model_dist_t = self._make_switch_transition_dist(
                lat_vars_tm1=lats_tm1.variables, ctrl_t=ctrl_t,
            )

        s_t = (
            switch_model_dist_t.mean
            if deterministic
            else switch_model_dist_t.sample()
        )
        gls_params_t = self.gls_base_parameters(switch=s_t, controls=ctrl_t,)

        x_dist_t = torch.distributions.MultivariateNormal(
            loc=(
                matvec(gls_params_t.A, lats_tm1.variables.x)
                if gls_params_t.A is not None
                else lats_tm1.variables.x
            )
            + (gls_params_t.b if gls_params_t.b is not None else 0.0),
            scale_tril=gls_params_t.LR,  # stable with scale and 0 variance.
        )

        x_t = x_dist_t.mean if deterministic else x_dist_t.sample()
        (m_t, V_t) = (None, None)
        # emission_dist = self.emit(lats_t=lats_t, ctrl_t=ctrl_t)
        emission_dist_t = torch.distributions.MultivariateNormal(
            loc=matvec(gls_params_t.C, x_t)
            + (gls_params_t.d if gls_params_t.d is not None else 0.0),
            scale_tril=gls_params_t.LQ,
        )
        emissions_t = (
            emission_dist_t.mean if deterministic else emission_dist_t.sample()
        )

        # NOTE: Should compute Cov if need forecast joint distribution.
        lats_t = LatentsSGLS(
            log_weights=lats_tm1.log_weights,  # does not change w/o evidence.
            gls_params=None,  # not used outside this function
            variables=GLSVariablesSGLS(
                x=x_t, m=m_t, V=V_t, Cov=None, switch=s_t,
            ),
        )
        return Prediction(latents=lats_t, emissions=emissions_t)

    def _sample_initial_latents(self, n_particle, n_batch,) -> LatentsSGLS:
        state_prior = self.state_prior_model(
            None, batch_shape_to_prepend=(n_particle, n_batch)
        )
        s_initial = None  # initial step has no switch sample.
        x_initial = state_prior.sample()
        (m, V) = (None, None)
        return LatentsSGLS(
            log_weights=torch.zeros_like(state_prior.loc[..., 0]),
            gls_params=None,  # initial step has none
            variables=GLSVariablesSGLS(
                x=x_initial, m=m, V=V, Cov=None, switch=s_initial,
            ),
        )

    def emit(
        self, lats_t: LatentsSGLS, ctrl_t: ControlInputsSGLS,
    ) -> torch.distributions.MultivariateNormal:
        # Unfortunately need to recompute gls_params.
        # Trade-off: faster, lower memory training vs. slower sampling/forecast
        gls_params_t = self.gls_base_parameters(
            switch=lats_t.variables.switch, controls=ctrl_t,
        )

        if lats_t.variables.m is not None:  # marginalise states
            mpy_t, Vpy_t = filter_forward_predictive_distribution(
                m=lats_t.variables.m,
                V=lats_t.variables.V,
                Q=gls_params_t.Q,
                C=gls_params_t.C,
                d=gls_params_t.d,
            )
            return MultivariateNormal(
                loc=mpy_t, scale_tril=cholesky(Vpy_t),
            )
        else:  # emit from state sample
            return torch.distributions.MultivariateNormal(
                loc=matvec(gls_params_t.C, lats_t.variables.x)
                + (gls_params_t.d if gls_params_t.d is not None else 0.0),
                scale_tril=gls_params_t.LQ,
            )

    def _make_encoder_dists(
        self, tar_t: torch.Tensor, ctrl_t: ControlInputsSGLS,
    ) -> Box:
        if self.encoder is None:
            return Box(switch=None)
        else:
            enc_inp = (
                [tar_t] if ctrl_t.encoder is None else [tar_t, ctrl_t.encoder]
            )
            encoded = self.encoder(enc_inp)
            if isinstance(encoded, torch.distributions.Distribution):
                return Box(switch=encoded)
            elif hasattr(encoded, "switch"):
                return encoded
            else:
                raise Exception(f"unknown encoding type: {type(encoded)}")

    def _make_switch_prior_dist(
        self,
        lat_vars_tm1: Optional[GLSVariablesSGLS],
        ctrl_t: ControlInputsSGLS,
        n_particle: int,
        n_batch: int,
    ) -> torch.distributions.MultivariateNormal:
        switch_model_dist = self.switch_prior_model(
            ctrl_t.switch,
            batch_shape_to_prepend=(n_particle,)
            + ((n_batch,) if ctrl_t.switch is None else ()),
        )
        return switch_model_dist

    def _make_switch_transition_dist(
        self,
        lat_vars_tm1: GLSVariablesSGLS,
        ctrl_t: Optional[ControlInputsSGLS],
    ) -> torch.distributions.MultivariateNormal:
        controls = (
            prepend_batch_dims(ctrl_t.switch, shp=(self.n_particle,))
            if (ctrl_t is not None and ctrl_t.switch is not None)
            else None
        )
        switch_model_dist = self.switch_transition_model(
            controls=controls, switch=lat_vars_tm1.switch,
        )
        return switch_model_dist

    def _make_switch_proposal_dist(
        self,
        switch_model_dist: torch.distributions.Distribution,
        switch_encoder_dist: Optional[torch.distributions.Distribution],
    ) -> torch.distributions.Distribution:
        if switch_encoder_dist is None:
            return switch_model_dist
        else:
            switch_proposal_dist = self.fuse_densities(
                [switch_model_dist, switch_encoder_dist]
            )
        return switch_proposal_dist

    # TODO: everything below code duplication.
    #  Currently quick solution only for re-producing pendulum plots
    def predict_marginals(
        self,
        # prediction_length would be misleading as prediction includes past.
        n_steps_forecast: int,
        past_targets: [Sequence[torch.Tensor], torch.Tensor],
        past_controls: Optional[
            Union[Sequence[ControlInputs], ControlInputs]
        ] = None,
        future_controls: Optional[Sequence[ControlInputs]] = None,
        deterministic: bool = False,
        smooth_past: bool = False,
    ) -> Tuple[Sequence[Prediction], Sequence[Prediction]]:  # past & future
        if smooth_past:
            latents_inferred = self.smooth(
                past_targets=past_targets, past_controls=past_controls,
            )
        else:
            latents_inferred = self.filter(
                past_targets=past_targets, past_controls=past_controls,
            )

        emission_dist_inferred = [
            self.emit(lats_t=latents_inferred[t], ctrl_t=past_controls[t])
            for t in range(len(latents_inferred))
        ]
        predictions_inferred = [
            Prediction(latents=l, emissions=e)
            for l, e in zip(latents_inferred, emission_dist_inferred)
        ]

        predictions_forecast = self.forecast_marginals(
            n_steps_forecast=n_steps_forecast,
            initial_latent=latents_inferred[-1],
            future_controls=future_controls,
            deterministic=deterministic,
        )
        return predictions_inferred, predictions_forecast

    def forecast_marginals(
        self,
        n_steps_forecast: int,
        initial_latent: LatentsSGLS,
        future_controls: Optional[
            Union[Sequence[ControlInputs], ControlInputs]
        ] = None,
        deterministic: bool = False,
    ) -> Sequence[Prediction]:

        initial_latent, future_controls = self._prepare_forecast(
            initial_latent=initial_latent,
            controls=future_controls,
            deterministic=deterministic,
        )

        return self._marginal_trajectory_from_initial(
            n_steps_forecast=n_steps_forecast,
            initial_latent=initial_latent,
            future_controls=future_controls,
            deterministic=deterministic,
        )

    def _marginal_trajectory_from_initial(
        self,
        n_steps_forecast: int,
        initial_latent: LatentsSGLS,
        future_controls: Optional[
            Union[Sequence[ControlInputs], ControlInputs]
        ] = None,
        deterministic: bool = False,
    ) -> Sequence[Prediction]:
        # initial_latent is considered t == -1

        if future_controls is not None:
            assert n_steps_forecast == len(future_controls)

        controls = (
            [None] * n_steps_forecast
            if future_controls is None
            else future_controls
        )
        samples = [None] * n_steps_forecast

        for t in range(n_steps_forecast):
            samples[t] = self.marginal_step(
                lats_tm1=samples[t - 1].latents if t > 0 else initial_latent,
                ctrl_t=controls[t],
                deterministic=deterministic,
            )
        return samples

    def marginal_step(
            self,
            lats_tm1: LatentsSGLS,
            ctrl_t: ControlInputsSGLS,
            deterministic: bool = False,
    ) -> Prediction:
        # TODO: duplication with sample_step. Requires refactoring.
        n_batch = lats_tm1.variables.m.shape[1]

        if lats_tm1.variables.switch is None:
            switch_model_dist_t = self._make_switch_prior_dist(
                n_particle=self.n_particle,
                n_batch=n_batch,
                lat_vars_tm1=lats_tm1.variables,
                ctrl_t=ctrl_t,
            )
        else:
            switch_model_dist_t = self._make_switch_transition_dist(
                lat_vars_tm1=lats_tm1.variables, ctrl_t=ctrl_t,
            )

        s_t = (
            switch_model_dist_t.mean
            if deterministic
            else switch_model_dist_t.sample()
        )
        gls_params_t = self.gls_base_parameters(switch=s_t, controls=ctrl_t,)

        x_t = None
        m_t, V_t = filter_forward_prediction_step(
            m=lats_tm1.variables.m,
            V=lats_tm1.variables.V,
            R=gls_params_t.R,
            A=gls_params_t.A,
            b=gls_params_t.b,
        )
        mpy_t, Vpy_t = filter_forward_predictive_distribution(
            m=m_t,
            V=V_t,
            Q=gls_params_t.Q,
            C=gls_params_t.C,
            d=gls_params_t.d,
        )
        emission_dist_t = torch.distributions.MultivariateNormal(
            loc=mpy_t,
            scale_tril=cholesky(Vpy_t),
        )

        # NOTE: Should compute Cov if need forecast joint distribution.
        lats_t = LatentsSGLS(
            log_weights=lats_tm1.log_weights,  # does not change w/o evidence.
            gls_params=None,  # not used outside this function
            variables=GLSVariablesSGLS(
                x=x_t, m=m_t, V=V_t, Cov=None, switch=s_t,
            ),
        )
        return Prediction(latents=lats_t, emissions=emission_dist_t)