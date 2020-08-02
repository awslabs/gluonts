from typing import Sequence, Optional
from dataclasses import dataclass
from box import Box

import torch
from torch import nn
from torch.distributions import OneHotCategorical, MultivariateNormal

from models_new_will_replace.base_gls import GLSVariables
from models_new_will_replace.base_amortized_gls import (
    LatentsRBSMC,
    BaseAmortizedGaussianLinearSystem,
    Prediction,
)
from models_new_will_replace.sgls_rbpf import (
    SwitchingGaussianLinearSystemBaseRBSMC,
    ControlInputsSGLS,
)


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
)
from torch_extensions.distributions.conditional_parametrised_distribution import (
    LadderParametrisedConditionalDistribution,
)
from models_new_will_replace.gls_parameters.gls_parameters import GLSParameters


# ***** SGLS *****
@dataclass
class GLSVariablesSGLS(GLSVariables):

    switch: torch.Tensor


@dataclass
class LatentsSGLS(LatentsRBSMC):

    variables: GLSVariablesSGLS
    log_weights: torch.Tensor

    def __post_init__(self):
        assert isinstance(self.variables, GLSVariablesSGLS)

@dataclass
class GLSVariablesASGLS(GLSVariablesSGLS):

    auxiliary: torch.Tensor


@dataclass
class LatentsASGLS(LatentsRBSMC):

    variables: GLSVariablesASGLS
    log_weights: torch.Tensor


class AuxiliarySwitchingGaussianLinearSystemRBSMC(
    SwitchingGaussianLinearSystemBaseRBSMC
):
    def __init__(
        self,
        n_state: int,
        n_target: int,
        n_ctrl_state: int,
        n_ctrl_target: int,
        n_particle: int,
        n_switch: int,
        gls_base_parameters: GLSParameters,
        measurement_model: nn.Module,
        encoder: LadderParametrisedConditionalDistribution,
        state_prior_model: ParametrisedMultivariateNormal,
        switch_prior_model: ParametrisedDistribution,
        switch_transition_model: nn.Module,
        resampling_criterion_fn=EffectiveSampleSizeResampleCriterion(0.5),
        resampling_indices_fn: callable = systematic_resampling_indices,
    ):
        super().__init__(
            n_state=n_state,
            n_target=n_target,
            n_ctrl_state=n_ctrl_state,
            n_ctrl_target=n_ctrl_target,
            n_particle=n_particle,
            n_switch=n_switch,
            gls_base_parameters=gls_base_parameters,
            encoder=encoder,
            state_prior_model=state_prior_model,
            switch_prior_model=switch_prior_model,
            switch_transition_model=switch_transition_model,
            resampling_criterion_fn=resampling_criterion_fn,
            resampling_indices_fn=resampling_indices_fn,
        )
        self.measurement_model = measurement_model

    def filter_step(
        self,
        lats_tm1: (LatentsASGLS, None),
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
            lats_tm1 = LatentsASGLS(
                log_weights=None,  # Not used. We use log_norm_weights instead.
                gls_params=None,  # First (previous) step no gls_params
                variables=GLSVariablesASGLS(
                    m=state_prior.loc,
                    V=state_prior.covariance_matrix,
                    Cov=None,
                    x=None,
                    switch=None,
                    auxiliary=None,
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
            lats_tm1 = LatentsASGLS(
                log_weights=None,  # Not used. We use log_norm_weights instead.
                gls_params=None,  # not used outside this function.
                variables=GLSVariablesASGLS(
                    **resampled_tensors, x=None, Cov=None
                ),
            )
            switch_model_dist = self._make_switch_transition_dist(
                lat_vars_tm1=lats_tm1.variables, ctrl_t=ctrl_t,
            )

        encoder_dists = self._make_encoder_dists(tar_t=tar_t, ctrl_t=ctrl_t)
        switch_proposal_dist = self._make_switch_proposal_dist(
            switch_model_dist=switch_model_dist,
            switch_encoder_dist=encoder_dists.switch,
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
        auxiliary_model_dist = self._make_auxiliary_model_dist(
            mp=mp, Vp=Vp, gls_params=gls_params_t,
        )
        auxiliary_proposal_dist = self._make_auxiliary_proposal_dist(
            auxiliary_model_dist=auxiliary_model_dist,
            auxiliary_encoder_dist=encoder_dists.auxiliary,
        )
        z_t = auxiliary_proposal_dist.rsample()
        m_t, V_t = filter_forward_measurement_step(
            y=z_t,
            m=mp,
            V=Vp,
            Q=gls_params_t.Q,
            C=gls_params_t.C,
            d=gls_params_t.d,
        )
        measurement_dist = self.measurement_model(z_t)
        log_update = (
            measurement_dist.log_prob(tar_t)
            + auxiliary_model_dist.log_prob(z_t)
            + switch_model_dist.log_prob(s_t)
            - switch_proposal_dist.log_prob(s_t)
            - auxiliary_proposal_dist.log_prob(z_t)
        )
        log_weights_t = log_norm_weights + log_update
        return LatentsASGLS(
            log_weights=log_weights_t,
            gls_params=None,  # not used outside this function
            variables=GLSVariablesASGLS(
                m=m_t, V=V_t, x=None, Cov=None, switch=s_t, auxiliary=z_t,
            ),
        )

    def sample_step(
        self,
        lats_tm1: LatentsASGLS,
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

        x_dist_t = torch.distributions.MultivariateNormal(
            loc=(
                    matvec(gls_params_t.A, lats_tm1.variables.x)
                    if gls_params_t.A is not None
                    else lats_tm1.variables.x
                )
                + (gls_params_t.b if gls_params_t.b is not None else 0.0),
            scale_tril=gls_params_t.LR,  # stable with scale and 0 variance.
        )

        x_t = x_dist_t.mean if deterministic else x_dist_t.rsample()

        z_dist_t = torch.distributions.MultivariateNormal(
            loc=matvec(gls_params_t.C, x_t)
            + (gls_params_t.d if gls_params_t.d is not None else 0.0),
            scale_tril=cholesky(gls_params_t.Q),
        )
        z_t = z_dist_t.mean if deterministic else z_dist_t.rsample()

        lats_t = LatentsASGLS(
            log_weights=lats_tm1.log_weights,  # does not change w/o evidence.
            gls_params=None,  # not used outside this function
            variables=GLSVariablesASGLS(
                x=x_t, m=None, V=None, Cov=None, switch=s_t, auxiliary=z_t,
            ),
        )
        emission_dist = self.emit(lats_t=lats_t, ctrl_t=ctrl_t)
        emissions_t = emission_dist.mean \
            if deterministic \
            else emission_dist.rsample()

        return Prediction(latents=lats_t, emissions=emissions_t)

    def _sample_initial_latents(
        self,
        n_particle,
        n_batch,
    ) -> LatentsASGLS:
        state_prior = self.state_prior_model(
            None, batch_shape_to_prepend=(n_particle, n_batch)
        )
        x_initial = state_prior.sample()
        s_initial = None  # initial step has no switch sample.\
        z_initial = None  # same here.
        return LatentsASGLS(
            log_weights=torch.zeros_like(state_prior.loc[..., 0]),
            gls_params=None,  # initial step has none
            variables=GLSVariablesASGLS(
                x=x_initial,
                m=None,
                V=None,
                Cov=None,
                switch=s_initial,
                auxiliary=z_initial,
            )
        )

    def emit(self, lats_t: LatentsASGLS, ctrl_t: ControlInputsSGLS):
        return self.measurement_model(lats_t.variables.auxiliary)

    def _make_encoder_dists(
        self, tar_t: torch.Tensor, ctrl_t: ControlInputsSGLS,
    ) -> Box:
        encoded = self.encoder([tar_t, ctrl_t.encoder])
        if not isinstance(encoded, Sequence):
            raise Exception(f"Expected sequence, got {type(encoded)}")
        if not len(encoded) == 2:
            raise Exception(f"Expected 2 encodings, got {len(encoded)}")

        return Box(auxiliary=encoded[0], switch=encoded[1])

    def _make_auxiliary_model_dist(
        self,
        mp: torch.Tensor,
        Vp: torch.Tensor,
        gls_params: Box,
    ):
        mpz, Vpz = filter_forward_predictive_distribution(
            m=mp, V=Vp, Q=gls_params.Q, C=gls_params.C, d=gls_params.d,
        )
        auxiliary_model_dist = MultivariateNormal(
            loc=mpz, scale_tril=cholesky(Vpz)
        )
        return auxiliary_model_dist

    def _make_auxiliary_proposal_dist(
        self,
        auxiliary_model_dist: torch.distributions.Distribution,
        auxiliary_encoder_dist: torch.distributions.Distribution,
    ):
        return self.fuse_densities(
            [auxiliary_model_dist, auxiliary_encoder_dist]
        )
