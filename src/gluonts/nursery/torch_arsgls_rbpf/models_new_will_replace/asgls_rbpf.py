from dataclasses import dataclass
from box import Box

import torch
from torch import nn
from torch.distributions import OneHotCategorical, MultivariateNormal

from models_new_will_replace.dynamical_system import GLSVariables, ControlInputs
from models_new_will_replace.base_rbsmc import (
    LatentsRBSMC,
    BaseRBSMC,
    Prediction,
)
from models_new_will_replace.sgls_rbpf import SwitchingGaussianLinearSystemRBSMC


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


# class A:
#     def foo(self, a: LatentsRBSMC) -> torch.Tensor:
#         return a.variables.m
#
#
# class B(A):
#     def foo(self, a: LatentsSGLS) -> torch.Tensor:
#         return a.variables.switch

# @dataclass
# class State:
# """
# Stores either (m, V) or samples or both from a MultivariateNormal.
# We use this instead of torch.distributions.MultivariateNormal in order
# to reduce overhead and increase performance. However,
# performance differences should be tested later, maybe can replace this.
# """
#
# m: (torch.Tensor, None) = None
# V: (torch.Tensor, None) = None
# x: (torch.Tensor, None) = None
#
# def __post_init__(self):
#     has_state_dist_params = tuple(
#         param is not None for param in (self.m, self.V)
#     )
#     if not len(set(has_state_dist_params)) == 1:
#         raise Exception("Provide either all or no distribution parameters")
#
#     has_state_sample = self.x is not None
#     if not (all(has_state_dist_params) or has_state_sample):
#         raise Exception("Provide at least either dist params or samples.")


# @dataclass
# class Latents:  # per-time-step
#     """ Base Template for Gaussian Linear Systems. """
#
#     state: State  # we call the latents of the GLS states.
#
#     @property
#     def state_name(self):  # if we rename, only need once.
#         return "state"
#
#     def as_flat_dict(self):
#         return dict(FlatDict(asdict(self)))
#         # state = latents.pop(self.state_name)
#         #
#         # for key, value in asdict(state).items():
#         #     latents.update(f"{self.state_name}.{key}")
#
#     def from_flat_dict(self, flat_dict: dict):
#         # TODO: This is really bad. cannot make classmethod due to state_name.
#         # TODO: currently handles only "state:*",
#         #  not arbitrary flat_dict representation.
#         state_keys = tuple(
#             key for key in flat_dict if key.startswith(f"{self.state_name}:")
#         )
#         state_dict = {
#             key.split(f"{self.state_name}:")[1]: flat_dict.pop(key)
#             for key in state_keys
#         }
#         print(state_dict)
#         self.__init__(state=State(**state_dict), **flat_dict)
#         return self


# @dataclass
# class LatentsRBSMC(Latents):
#     """ Template for models based on Rao-Blackwellized SMC. """
#
#     log_weights: torch.Tensor
#
#
# @dataclass
# class LatentsSGLS(LatentsRBSMC):
#     """ Template for (Recurrent) Switching Gaussian Linear Systems. """
#
#     switch: torch.Tensor
#
#
# @dataclass
# class LatentASGLS(LatentsRBSMC):
#     """ Template for Auxiliary Variable (Recurrent) SGLS. """
#
#     switch: torch.Tensor
#     auxiliary: torch.Tensor


@dataclass
class GLSVariablesASGLS(GLSVariables):

    switch: torch.Tensor
    auxiliary: torch.Tensor


@dataclass
class LatentsASGLS(LatentsRBSMC):

    variables: GLSVariablesASGLS
    log_weights: torch.Tensor


class AuxiliarySwitchingGaussianLinearSystemRBSMC(SwitchingGaussianLinearSystemRBSMC):
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
        obs_encoder: nn.Module,
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
            n_switch=n_switch,
            gls_base_parameters=gls_base_parameters,
            obs_encoder=obs_encoder,
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
        ctrl_t: ControlInputs,
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
            lats_tm1 = LatentsASGLS(
                log_weights=None,  # Not used. We use log_norm_weights instead.
                variables=GLSVariablesASGLS(
                    m=state_prior.loc,
                    V=state_prior.covariance_matrix,
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
                    if key != "x"  # TODO: Is there a better way? Also below.
                },
                resampling_indices_fn=self.resampling_indices_fn,
                criterion_fn=self.resampling_criterion_fn,
            )
            lats_tm1 = LatentsASGLS(
                log_weights=None,  # Not used. We use log_norm_weights instead.
                variables=GLSVariablesASGLS(**resampled_tensors, x=None),
            )
            switch_model_dist = self._make_switch_transition_dist(
                lat_vars_tm1=lats_tm1.variables, ctrl_t=ctrl_t,
            )

        encoder_dists = self._make_encoder_dists(tar_t=tar_t, ctrl_t=ctrl_t,)
        switch_proposal_dist = self._make_switch_proposal_dist(
            switch_model_dist=switch_model_dist,
            switch_encoder_dist=encoder_dists.switch,
        )
        s_t = switch_proposal_dist.rsample()
        # TODO: change API also of gls_params! take ctrl.
        #  not all got seasonality stuff! maybe subclass will have other.
        gls_params_t = self.gls_base_parameters(
            switch=s_t,
            seasonal_indicators=None,  # TODO: should not have this. but will be resolved by API change
            u_state=ctrl_t.state,
            u_obs=ctrl_t.target,
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
            variables=GLSVariablesASGLS(
                m=m_t, V=V_t, x=None, switch=s_t, auxiliary=z_t,
            ),
        )

    def forecast_sample_step(
        self,
        lats_tm1: LatentsASGLS,
        ctrl_t: ControlInputs,
        deterministic: bool = False,
    ) -> Prediction:
        n_batch = lats_tm1.variables.x.shape[1]

        # TODO: Now we have again the distinction here. Can be improved?
        # TODO: Can use base cls, but API is different :-/
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
            seasonal_indicators=None,  # TODO: should not have this. but will be resolved by API change
            u_state=ctrl_t.state,
            u_obs=ctrl_t.target,
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

        z_dist_t = torch.distributions.MultivariateNormal(
            loc=matvec(gls_params_t.C, x_t)
            + (gls_params_t.d if gls_params_t.d is not None else 0.0),
            scale_tril=cholesky(gls_params_t.Q),
        )
        z_t = z_dist_t.mean if deterministic else z_dist_t.rsample()

        lats_t = LatentsASGLS(
            log_weights=lats_tm1.log_weights,  # does not change w/o evidence.
            variables=GLSVariablesASGLS(
                x=x_t, m=None, V=None, switch=s_t, auxiliary=z_t,
            ),
        )
        emission_dist = self.emit(lats_t=lats_t, ctrl_t=ctrl_t)
        emissions_t = emission_dist.mean \
            if deterministic \
            else emission_dist.rsample()

        return Prediction(latents=lats_t, emissions=emissions_t)

    def _sample_initial_latents(self, n_particle, n_batch) -> LatentsASGLS:
        state_prior = self.state_prior_model(
            None, batch_shape_to_prepend=(n_particle, n_batch)
        )
        x_initial = state_prior.sample()
        s_initial = None  # initial step has no switch sample.\
        z_initial = None  # same here.
        return LatentsASGLS(
            log_weights=torch.zeros_like(state_prior.loc[..., 0]),
            variables=GLSVariablesASGLS(
                x=x_initial,
                m=None,
                V=None,
                switch=s_initial,
                auxiliary=z_initial,
            )
        )

    def emit(self, lats_t: LatentsASGLS, ctrl_t: ControlInputs):
        return self.measurement_model(lats_t.variables.auxiliary)

    def _make_auxiliary_model_dist(
        self,
        mp: torch.Tensor,
        Vp: torch.Tensor,
        gls_params: Box,  # TODO: define a type for this.
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

    # def _initial_state(
    #     self,
    #     ctrl_initial: (ControlInputs, None),
    #     n_particle: int,
    #     n_batch: int,
    # ) -> LatentsASGLS:
    #     state_prior = self.state_prior_model(
    #         None, batch_shape_to_prepend=(n_particle, n_batch),
    #     )
    #     latent_variables_initial = LatentsASGLS(
    #         variables=RandomVariablesASGLS(
    #             m=state_prior.loc,
    #             V=state_prior.covariance_matrix,
    #             x=None,
    #             switch=None,
    #             auxiliary=None,
    #         ),
    #         log_weights=torch.zeros_like(state_prior.loc[..., 0]),
    #     )
    #     return latent_variables_initial

# class ArsglsGtsUnivariate(LightningModule):
#     def __init__(
#         self,
#         ssm: AuxiliarySwitchingGaussianLinearSystemRBSMC,
#         ctrl_transformer: nn.Module,  # TODO: *args, **kwargs -> ControlInputs
#         tar_transformer: torch.distributions.Transform,  # invertible
#     ):
#         self.ctrl_transformer = ctrl_transformer
#         self.tar_transformer = tar_transformer
#         self.ssm = ssm
#
#     def forward(self, observations, controls):
#         # TODO: API change: make inputs as in GluonTS and transformer handle.
#         #  Maybe implement base class for those GTS experiments.
#         #  Some private fns that do the pre-processing.
#         ctrls_ssm = self.ctrl_transformer(controls)
#         obs_ssm = self.tar_transformer(observations)
#         filtered = self.ssm.filter(observations=obs_ssm, controls=ctrls_ssm)
#         fil
