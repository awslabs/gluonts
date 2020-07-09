from typing import Optional
from dataclasses import dataclass

from box import Box

import torch
from torch import nn
from torch.distributions import MultivariateNormal

from models_new_will_replace.base_rbsmc import (
    LatentsRBSMC,
    BaseRBSMC,
    Prediction,
)
from models_new_will_replace.dynamical_system import GLSVariables, ControlInputs

from inference.smc.resampling import (
    resample,
    make_criterion_fn_with_ess_threshold,
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
from models_new_will_replace.gls_parameters.gls_parameters import GLSParameters

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


class SwitchingGaussianLinearSystemRBSMC(BaseRBSMC):
    def __init__(
        self,
        n_state: int,
        n_target: int,
        n_ctrl_state: int,
        n_ctrl_target: int,
        n_particle: int,
        n_switch: int,
        gls_base_parameters: GLSParameters,
        obs_encoder: ParametrisedConditionalDistribution,
        state_prior_model: ParametrisedMultivariateNormal,
        switch_prior_model: ParametrisedDistribution,
        switch_transition_model: nn.Module,
        resampling_criterion_fn=make_criterion_fn_with_ess_threshold(0.5),
        resampling_indices_fn: callable = systematic_resampling_indices,
    ):
        super().__init__(
            n_state=n_state,
            n_target=n_target,
            n_ctrl_state=n_ctrl_state,
            n_ctrl_target=n_ctrl_target,
            n_particle=n_particle,
            gls_base_parameters=gls_base_parameters,
            obs_encoder=obs_encoder,
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
    ):
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
                variables=GLSVariablesSGLS(
                    m=state_prior.loc,
                    V=state_prior.covariance_matrix,
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
                    if key != "x"  # below set to None explicitly
                },
                resampling_indices_fn=self.resampling_indices_fn,
                criterion_fn=self.resampling_criterion_fn,
            )
            lats_tm1 = LatentsSGLS(
                log_weights=None,  # Not used. We use log_norm_weights instead.
                variables=GLSVariablesSGLS(**resampled_tensors, x=None),
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
        gls_params_t = self.gls_base_parameters(
            switch=s_t,
            controls=ctrl_t,
        )

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
            m=mp,
            V=Vp,
            Q=gls_params_t.Q,
            C=gls_params_t.C,
            d=gls_params_t.d,
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
            variables=GLSVariablesSGLS(
                m=m_t, V=V_t, x=None, switch=s_t,
            ),
        )

    def forecast_sample_step(
        self,
        lats_tm1: LatentsSGLS,
        ctrl_t: ControlInputsSGLS,
        deterministic: bool = False,
    ) -> Prediction:
        n_batch = lats_tm1.variables.x.shape[1]

        # TODO: Now we have again the distinction here. Can be improved?
        if lats_tm1.variables.switch is None:
            switch_model_dist_t = self._make_switch_prior_dist(
                n_particle=self.n_particle,
                n_batch=n_batch,
                lat_vars_tm1=lats_tm1.variables,
                ctrl_t=ctrl_t,
            )
        else:
            switch_model_dist_t = self._make_switch_transition_dist(
                lat_vars_tm1=lats_tm1.variables,
                ctrl_t=ctrl_t,
            )

        s_t = (
            switch_model_dist_t.mean
            if deterministic
            else switch_model_dist_t.rsample()
        )
        gls_params_t = self.gls_base_parameters(
            switch=s_t,
            controls=ctrl_t,
        )

        # covs are not psd in case of ISSM (zeros on most entries).
        # fortunately, these are diagonal -> don't need cholesky, just sqrt of diag.
        # TODO: maybe extract short naming in the beginning...
        # TODO: MultivariateNormal exists also in cholesky form in torch I think.
        #  can that solve this annoying issue? OR just IndependentNormal?
        #  Although states are mixed (full cov) after prediction/update step.
        try:
            x_dist_t = torch.distributions.MultivariateNormal(
                loc=(
                    matvec(gls_params_t.A, lats_tm1.variables.x)
                    if gls_params_t.A is not None
                    else lats_tm1.variables.x
                )
                + (gls_params_t.b if gls_params_t.b is not None else 0.0),
                covariance_matrix=gls_params_t.R,
            )
        except:
            assert (
                batch_diag_matrix(batch_diag(gls_params_t.R)) == gls_params_t.R
            ).all()
            x_dist_t = torch.distributions.MultivariateNormal(
                loc=(
                    matvec(gls_params_t.A, lats_tm1.variables.x)
                    if gls_params_t.A is not None
                    else lats_tm1.variables.x
                )
                + (gls_params_t.b if gls_params_t.b is not None else 0.0),
                scale_tril=batch_diag_matrix(
                    batch_diag(gls_params_t.R) ** 0.5
                ),
            )

        x_t = x_dist_t.mean if deterministic else x_dist_t.rsample()
        lats_t = LatentsSGLS(
            log_weights=lats_tm1.log_weights,  # does not change w/o evidence.
            variables=GLSVariablesSGLS(
                x=x_t, m=None, V=None, switch=s_t,
            ),
        )

        # emission_dist = self.emit(lats_t=lats_t, ctrl_t=ctrl_t)
        emission_dist_t = torch.distributions.MultivariateNormal(
            loc=matvec(gls_params_t.C, x_t)
                + (gls_params_t.d if gls_params_t.d is not None else 0.0),
            scale_tril=cholesky(gls_params_t.Q),
        )
        emissions_t = emission_dist_t.mean \
            if deterministic \
            else emission_dist_t.rsample()

        return Prediction(latents=lats_t, emissions=emissions_t)

    def _sample_initial_latents(self, n_particle, n_batch) -> LatentsSGLS:
        state_prior = self.state_prior_model(
            None, batch_shape_to_prepend=(n_particle, n_batch)
        )
        x_initial = state_prior.sample()
        s_initial = None  # initial step has no switch sample.
        return LatentsSGLS(
            log_weights=torch.zeros_like(state_prior.loc[..., 0]),
            variables=GLSVariablesSGLS(
                x=x_initial,
                m=None,
                V=None,
                switch=s_initial,
            )
        )

    def emit(self, lats_t: LatentsSGLS, ctrl_t: ControlInputsSGLS):
        # Unfortunately need to recompute gls_params.
        # Trade-off: faster, lower memory training vs. slower sampling/forecast
        gls_params_t = self.gls_base_parameters(
            switch=lats_t.variables.switch,
            controls=ctrl_t,
        )
        return torch.distributions.MultivariateNormal(
            loc=matvec(gls_params_t.C, lats_t.variables.x)
                + (gls_params_t.d if gls_params_t.d is not None else 0.0),
            scale_tril=cholesky(gls_params_t.Q),
        )

    def _make_encoder_dists(
        self, tar_t: torch.Tensor, ctrl_t: ControlInputsSGLS,
    ) -> Box:
        encoded = self.obs_encoder([tar_t, ctrl_t.encoder])
        if isinstance(encoded, torch.distributions.Distribution):
            return Box(switch=encoded)
        elif hasattr(encoded, "switch"):
            return encoded.switch
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
        self, lat_vars_tm1: GLSVariablesSGLS, ctrl_t: ControlInputsSGLS,
    ) -> torch.distributions.MultivariateNormal:
        # TODO: make base class and eventually move the prepend in.
        switch_model_dist = self.switch_transition_model(
            u=prepend_batch_dims(ctrl_t.switch, shp=(self.n_particle,))
            if ctrl_t.switch is not None
            else None,
            s=lat_vars_tm1.switch,
        )
        return switch_model_dist

    def _make_switch_proposal_dist(
        self,
        switch_model_dist: torch.distributions.Distribution,
        switch_encoder_dist: torch.distributions.Distribution,
    ) -> torch.distributions.MultivariateNormal:
        switch_proposal_dist = self.fuse_densities(
            [switch_model_dist, switch_encoder_dist]
        )
        return switch_proposal_dist


