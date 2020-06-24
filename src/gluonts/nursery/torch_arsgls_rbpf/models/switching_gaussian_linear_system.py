from typing import Iterator, Optional, Dict
from box import Box

import numpy as np
import torch
from torch import nn
from torch.distributions import OneHotCategorical, MultivariateNormal

from gluonts.dataset.common import Dataset
from gluonts.dataset.field_names import FieldName
from gluonts.model.forecast import Forecast, SampleForecast
from gluonts.model.predictor import RepresentablePredictor
from gluonts.dataset.loader import InferenceDataLoader

from utils.utils import (
    TensorDims,
    create_zeros_log_weights,
    list_of_dicts_to_dict_of_list,
)
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
    vectorize_normal_dist_params,
)
from torch_extensions.fusion import ProbabilisticSensorFusion
from torch_extensions.distributions.parametrised_distribution import (
    ParametrisedDistribution,
    ParametrisedMultivariateNormal,
    prepend_batch_dims,
)
from torch_extensions.distributions.stable_relaxed_categorical import (
    StableRelaxedOneHotCategorical,
)
from models.gls_parameters import GLSParameters
from models.dynamical_system import DynamicalSystem
from experiments.model_component_zoo.input_transforms import InputTransformer
from data.gluonts_nips_datasets.gluonts_nips_datasets import (
    transform_gluonts_to_pytorch,
)


class SwitchingLinearDynamicalSystem(DynamicalSystem):
    def __init__(
        self,
        n_state: int,
        n_obs: int,
        n_ctrl_state: int,
        n_particle: int,
        n_switch: int,
        gls_base_parameters: GLSParameters,
        input_transformer: (InputTransformer, None),
        state_prior_model: ParametrisedMultivariateNormal,
        switch_prior_model: ParametrisedDistribution,
        switch_transition_model: nn.Module,
        obs_to_switch_encoder: (nn.Module, None),
        state_to_switch_encoder: (nn.Module, None),
        min_ess_ratio: (float, None) = 0.5,
        resampling_indices_fn: callable = systematic_resampling_indices,
    ):
        super().__init__(
            n_state=n_state,
            n_obs=n_obs,
            n_ctrl_state=n_ctrl_state,
            n_particle=n_particle,
        )
        self.n_switch = n_switch
        self.gls_base_parameters = gls_base_parameters
        self.input_transformer = input_transformer
        self.switch_transition_model = switch_transition_model
        self.obs_to_switch_encoder = obs_to_switch_encoder
        self.state_to_switch_encoder = state_to_switch_encoder
        self.fuse_densities = ProbabilisticSensorFusion()
        self.state_prior_model = state_prior_model
        self.switch_prior_model = switch_prior_model
        self.resampling_criterion_fn = make_criterion_fn_with_ess_threshold(
            min_ess_ratio=min_ess_ratio
        )
        self.resampling_indices_fn = resampling_indices_fn

    def get_dims(self, *args, **kwargs):
        dims = super().get_dims(*args, **kwargs)._asdict()
        dims.update(switch=self.n_switch)
        dims = TensorDims(**dims)
        return dims

    def forward(
        self,
        y,
        u_static_cat=None,
        u_time=None,
        seasonal_indicators=None,
        rsample=True,
        make_forecast=False,
        n_steps_forecast=None,
    ):
        # TODO: resolve this ugliness!!!
        #  (DataParallel only wraps the forward method...
        #  But there are better ways to do this. Maybe only call another method)
        #  This is now basically loss + filter_sample.
        #  filter_forecast not used because it computes marginals
        #  (more useful for plotting, not for sampling).

        # TODO: computes currently loss time-batch-wise. Need only batch-wise.

        if make_forecast:
            assert isinstance(n_steps_forecast, int)
            dims = self.get_dims(y=y)
            n_steps_forecast = n_steps_forecast
            n_steps_filter, n_batch = dims.timesteps, dims.batch
            n_steps_total = n_steps_filter + n_steps_forecast
            del dims
            assert all(
                len(u) == n_steps_total
                for u in [u_static_cat, u_time, seasonal_indicators]
                if u is not None
            )

            filter_results = self.filter_forward(
                y=y[:n_steps_filter],
                seasonal_indicators=seasonal_indicators[:n_steps_filter]
                if seasonal_indicators is not None
                else None,
                u_static_cat=u_static_cat[:n_steps_filter]
                if u_static_cat is not None
                else None,
                u_time=u_time[:n_steps_filter] if u_time is not None else None,
                rsample=rsample,
            )
            (
                log_weights_filter,
                s_filter,
                m_filter,
                V_filter,
                mpy_filter,
                Vpy_filter,
                gls_params_filter,
            ) = filter_results

            # Resample
            log_weights, s, m, V = (
                log_weights_filter[-1],
                s_filter[-1],
                m_filter[-1],
                V_filter[-1],
            )
            log_norm_weights = normalize_log_weights(log_weights=log_weights)
            log_norm_weights, (s, m, V) = resample(
                n_particle=self.n_particle,
                log_norm_weights=log_norm_weights,
                tensors_to_resample=(s, m, V),
                resampling_indices_fn=self.resampling_indices_fn,
                criterion_fn=make_criterion_fn_with_ess_threshold(
                    min_ess_ratio=1.0
                ),  # always
            )
            # Sample from last state
            x = MultivariateNormal(loc=m, scale_tril=cholesky(V)).sample()

            # Sample from generative model forward
            (
                s_forecast,
                x_forecast,
                y_forecast,
                gls_params_forecast,
            ) = self.sample(
                s=s,
                x=x,  # single (last filter) time-step.
                u_static_cat=u_static_cat[n_steps_filter:]
                if u_static_cat is not None
                else None,
                u_time=u_time[n_steps_filter:] if u_time is not None else None,
                seasonal_indicators=seasonal_indicators[n_steps_filter:]
                if seasonal_indicators is not None
                else None,
                n_timesteps=n_steps_forecast,
                n_batch=n_batch,
            )
            log_conditional_likelihoods_batchwise = tuple(
                torch.logsumexp(lws, dim=0) for lws in log_weights_filter
            )
            log_marginal_likelihood = torch.stack(
                log_conditional_likelihoods_batchwise, dim=0
            )
            losses_time_batch_wise = -log_marginal_likelihood
            return losses_time_batch_wise, y_forecast, mpy_filter, Vpy_filter
        else:
            return self.loss_forward(
                y=y,
                u_static_cat=u_static_cat,
                u_time=u_time,
                seasonal_indicators=seasonal_indicators,
                rsample=rsample,
            )

    def loss_forward(
        self,
        y,
        u_static_cat=None,
        u_time=None,
        seasonal_indicators=None,
        rsample=True,
    ):
        """ computes loss TB-wise """
        log_weights, s, m, V, mpy, Vpy, gls_params = self.filter_forward(
            y=y,
            u_static_cat=u_static_cat,
            u_time=u_time,
            seasonal_indicators=seasonal_indicators,
            rsample=rsample,
        )
        # sum over batches and  LSE over particle dim.
        log_conditional_likelihoods_batchwise = torch.stack(
            [torch.logsumexp(lws, dim=0) for lws in log_weights], dim=0
        )
        return -log_conditional_likelihoods_batchwise

    def filter_forecast(
        self,
        n_steps_forecast,
        y,
        u_time=None,
        u_static_cat=None,
        seasonal_indicators=None,
    ):
        # TODO: Make this function compute the whole joint - if it is possible.
        #  Otherwise this function is useful only for plotting...
        dims = self.get_dims(y=y)
        n_steps_filter, n_batch = dims.timesteps, dims.batch
        n_steps_total = n_steps_filter + n_steps_forecast

        assert all(
            len(u) == n_steps_total
            for u in [u_time, u_static_cat, seasonal_indicators]
            if u is not None
        )

        # Filter
        (
            log_weights_filter,
            s_filter,
            m_filter,
            V_filter,
            mpy_filter,
            Vpy_filter,
            gls_params_filter,
        ) = self.filter_forward(
            y=y[:n_steps_filter],
            seasonal_indicators=seasonal_indicators[:n_steps_filter]
            if seasonal_indicators is not None
            else None,
            u_static_cat=u_static_cat[:n_steps_filter]
            if u_static_cat is not None
            else None,
            u_time=u_time[:n_steps_filter] if u_time is not None else None,
        )
        # Resample
        log_weights, s, m, V = (
            log_weights_filter[-1],
            s_filter[-1],
            m_filter[-1],
            V_filter[-1],
        )
        log_norm_weights = normalize_log_weights(log_weights=log_weights)
        log_norm_weights, (s, m, V) = resample(
            n_particle=dims.particle,
            log_norm_weights=log_norm_weights,
            tensors_to_resample=(s, m, V),
            resampling_indices_fn=self.resampling_indices_fn,
            criterion_fn=make_criterion_fn_with_ess_threshold(
                min_ess_ratio=1.0
            ),  # always
        )
        # Forecast
        (
            s_forecast,
            m_forecast,
            V_forecast,
            _,
            _,
            gls_params_forecast,
        ) = self.forecast(
            s=s,
            m=m,
            V=V,  # single (last filter) time-step.
            u_static_cat=u_static_cat[n_steps_filter:]
            if u_static_cat is not None
            else None,
            u_time=u_time[n_steps_filter:] if u_time is not None else None,
            seasonal_indicators=seasonal_indicators[n_steps_filter:]
            if seasonal_indicators is not None
            else None,
            n_timesteps=n_steps_forecast,
            n_batch=n_batch,
        )
        # Concatenate filter and forecast
        assert all(
            [
                isinstance(item, (list, tuple, set))
                for item in [
                    s_filter,
                    s_forecast,
                    m_filter,
                    m_forecast,
                    V_filter,
                    V_forecast,
                ]
            ]
        )
        s_trajectory = torch.stack(tuple(s_filter) + tuple(s_forecast), dim=0)
        m_trajectory = torch.stack(tuple(m_filter) + tuple(m_forecast), dim=0)
        V_trajectory = torch.stack(tuple(V_filter) + tuple(V_forecast), dim=0)

        log_norm_weights_filter = normalize_log_weights(
            log_weights=torch.stack(log_weights_filter, dim=0), dim=-2
        )
        log_norm_weights_forecast = log_norm_weights[None, ...].repeat(
            n_steps_forecast, *(1,) * log_norm_weights.ndim
        )
        log_norm_weights_trajectory = torch.cat(
            (log_norm_weights_filter, log_norm_weights_forecast), dim=0
        )
        gls_params_trajectory = Box()
        for name in gls_params_filter.keys():
            if gls_params_filter[name] is not None and not all(
                param is None for param in gls_params_filter[name]
            ):
                gls_params_trajectory[name] = torch.stack(
                    tuple(gls_params_filter[name])
                    + tuple(gls_params_forecast[name]),
                    dim=0,
                )
            else:
                gls_params_trajectory[name] = None
        # Predictive distribution
        predictive_dist_params = [
            filter_forward_predictive_distribution(
                m=m_trajectory[t],
                V=V_trajectory[t],
                Q=gls_params_trajectory.Q[t],
                C=gls_params_trajectory.C[t],
                d=gls_params_trajectory.d[t]
                if gls_params_trajectory.d is not None
                else None,
            )
            for t in range(n_steps_total)
        ]
        mpy_trajectory = torch.stack(
            tuple(predictive_dist_params[t][0] for t in range(n_steps_total)),
            dim=0,
        )
        Vpy_trajectory = torch.stack(
            tuple(predictive_dist_params[t][1] for t in range(n_steps_total)),
            dim=0,
        )

        return (
            log_norm_weights_trajectory,
            s_trajectory,
            m_trajectory,
            V_trajectory,
            mpy_trajectory,
            Vpy_trajectory,
            gls_params_trajectory,
        )

    def filter_forward(
        self,
        y,
        u_static_cat=None,
        u_time=None,
        seasonal_indicators=None,
        rsample=True,
    ):
        u = self.input_transformer(u_static_cat=u_static_cat, u_time=u_time)
        dims = self.get_dims(y=y)
        dtype, device = (
            self.state_prior_model.m.dtype,
            self.state_prior_model.m.device,
        )

        u_state, u_obs, u_switch = u.state, u.obs, u.state
        (u_state, u_obs, u_switch, seasonal_indicators) = (
            [None] * dims.timesteps if u is None else u
            for u in (u_state, u_obs, u_switch, seasonal_indicators)
        )

        log_weights, m, V, mpy, Vpy, s, gls_params = (
            [None] * dims.timesteps for _ in range(7)
        )

        for t in range(dims.timesteps):
            if t == 0:
                state_prior = self.state_prior_model(
                    None, batch_shape_to_prepend=(dims.particle, dims.batch)
                )
                (
                    log_weights[t],
                    s[t],
                    m[t],
                    V[t],
                    mpy[t],
                    Vpy[t],
                    gls_params[t],
                ) = self._filter_forward_step(
                    dims=dims,
                    rsample=rsample,
                    log_weights=create_zeros_log_weights(
                        dims=dims, device=device, dtype=dtype
                    )[0],
                    s=None,
                    m=state_prior.loc,
                    V=state_prior.covariance_matrix,
                    y=y[t],
                    u_state=u_state[t],
                    u_obs=u_obs[t],
                    u_switch=u_switch[t],
                    seasonal_indicators=seasonal_indicators[t],
                )
            else:
                (
                    log_weights[t],
                    s[t],
                    m[t],
                    V[t],
                    mpy[t],
                    Vpy[t],
                    gls_params[t],
                ) = self._filter_forward_step(
                    dims=dims,
                    rsample=rsample,
                    log_weights=log_weights[t - 1],
                    s=s[t - 1],
                    m=m[t - 1],
                    V=V[t - 1],
                    y=y[t],
                    u_state=u_state[t],
                    u_obs=u_obs[t],
                    u_switch=u_switch[t],
                    seasonal_indicators=seasonal_indicators[t],
                )
        gls_params = list_of_dicts_to_dict_of_list(gls_params)
        return log_weights, s, m, V, mpy, Vpy, gls_params

    def forecast(
        self,
        s,
        m,
        V,
        u_static_cat=None,
        u_time=None,
        seasonal_indicators=None,
        n_timesteps=None,
        n_batch=None,
    ):
        """
        Unrolls the model by alternating
        1) sampling switch,
        2) prediction step state,
        3) computing the predictive distribution for observations.

        Note that the resulting switch trajectory is a sample from the joint distribution,
        whereas the state parameters (m, V) and the predictive distribution parameters (mpy, VPy)
        are marginal distributions (conditioned on the switch samples).
        """
        assert (
            len(set(tensor is None for tensor in (s, m, V))) == 1
        ), "all of (s, m, V) or none should be provided."
        initial_provided = s is not None
        u = self.input_transformer(u_static_cat=u_static_cat, u_time=u_time)
        u_state, u_obs, u_switch = u.state, u.obs, u.state
        dims = self.get_dims(
            u_state=u_state,
            u_obs=u_obs,
            n_timesteps=n_timesteps,
            n_batch=n_batch,
        )

        if initial_provided:
            (s_initial, m_initial, V_initial) = (s, m, V)
        else:
            # We have "x_{-1}" but not "s_{-1}". There is one more x than s in graphical model.
            # We must provide the prior for x; "None" for s is handled in make_switch_model_dist.
            state_prior = self.state_prior_model(
                None, batch_shape_to_prepend=(dims.particle, dims.batch)
            )
            (m_initial, V_initial) = (
                state_prior.loc,
                state_prior.covariance_matrix,
            )
            s_initial = None
        (u_state, u_obs, u_switch, seasonal_indicators) = (
            [None] * dims.timesteps if u is None else u
            for u in (u_state, u_obs, u_switch, seasonal_indicators)
        )
        (s, m, V, mpy, Vpy, gls_params) = (
            [None] * dims.timesteps for _ in range(6)
        )

        for t in range(n_timesteps):
            (
                s[t],
                m[t],
                V[t],
                mpy[t],
                Vpy[t],
                gls_params[t],
            ) = self._forecast_step(
                dims=dims,
                s=s[t - 1] if t > 0 else s_initial,
                m=m[t - 1] if t > 0 else m_initial,
                V=V[t - 1] if t > 0 else V_initial,
                u_state=u_state[t],
                u_obs=u_obs[t],
                u_switch=u_switch[t],
                seasonal_indicators=seasonal_indicators[t],
            )
        return s, m, V, mpy, Vpy, list_of_dicts_to_dict_of_list(gls_params)

    def sample(
        self,
        s=None,
        x=None,
        u_static_cat=None,
        u_time=None,
        seasonal_indicators=None,
        n_timesteps=None,
        n_batch=None,
    ):
        assert (
            len(set(tensor is None for tensor in (s, x))) == 1
        ), "all of (s, x) or none should be provided."
        u = self.input_transformer(u_static_cat=u_static_cat, u_time=u_time)
        u_state, u_obs, u_switch = u.state, u.obs, u.state
        dims = self.get_dims(
            u_state=u_state,
            u_obs=u_obs,
            n_timesteps=n_timesteps,
            n_batch=n_batch,
        )
        initial_provided = s is not None
        if initial_provided:
            (s_initial, x_initial) = (s, x)
        else:
            # We have "x_{-1}" but not "s_{-1}". There is one more x than s in graphical model.
            # We must provide sample of x; "None" for s is handled in make_switch_model_dist.
            state_prior = self.state_prior_model(
                None, batch_shape_to_prepend=(dims.particle, dims.batch)
            )
            x_initial = state_prior.sample()
            s_initial = None
        (u_state, u_obs, u_switch, seasonal_indicators) = (
            [None] * dims.timesteps if u is None else u
            for u in (u_state, u_obs, u_switch, seasonal_indicators)
        )
        (s, x, y, gls_params) = ([None] * dims.timesteps for _ in range(4))

        for t in range(n_timesteps):
            s[t], x[t], y[t], gls_params[t] = self._sample_step(
                dims=dims,
                s=s[t - 1] if t > 0 else s_initial,
                x=x[t - 1] if t > 0 else x_initial,
                seasonal_indicators=seasonal_indicators[t],
                u_state=u_state[t],
                u_obs=u_obs[t],
                u_switch=u_switch[t],
            )

        return s, x, y, list_of_dicts_to_dict_of_list(gls_params)

    def _filter_forward_step(
        self,
        dims,
        log_weights,
        s,
        m,
        V,
        y,
        u_state=None,
        u_obs=None,
        u_switch=None,
        seasonal_indicators=None,
        rsample=True,
    ):
        log_norm_weights = normalize_log_weights(log_weights=log_weights)
        if s is not None:
            log_norm_weights, (s, m, V) = resample(
                n_particle=dims.particle,
                log_norm_weights=log_norm_weights,
                tensors_to_resample=(s, m, V),
                resampling_indices_fn=self.resampling_indices_fn,
                criterion_fn=self.resampling_criterion_fn,
            )
        switch_model_dist = self._make_switch_model_dist(
            dims=dims, s=s, u_switch=u_switch, m=m, V=V,
        )
        switch_proposal_dist = self._make_switch_proposal_dist(
            switch_model_dist=switch_model_dist,
            # Fuse model dist with encoder dist
            m=m,
            V=V,
            y=y,
            u_state=u_state,
            u_obs=u_obs,
            u_switch=u_switch,
        )
        s_tp1 = (
            switch_proposal_dist.rsample()
            if rsample
            else switch_proposal_dist.sample()
        )
        gls_params_tp1 = self.gls_base_parameters(
            switch=s_tp1,
            seasonal_indicators=seasonal_indicators,
            u_state=u_state,
            u_obs=u_obs,
        )
        mp, Vp = filter_forward_prediction_step(
            m=m,
            V=V,
            R=gls_params_tp1.R,
            A=gls_params_tp1.A,
            b=gls_params_tp1.b,
        )
        # assert batch_diag(Vp).min() > 0
        m_tp1, V_tp1 = filter_forward_measurement_step(
            y=y,
            m=mp,
            V=Vp,
            Q=gls_params_tp1.Q,
            C=gls_params_tp1.C,
            d=gls_params_tp1.d,
        )
        # assert batch_diag(V_tp1).min() > 0
        mpy_tp1, Vpy_tp1 = filter_forward_predictive_distribution(
            m=mp,
            V=Vp,
            Q=gls_params_tp1.Q,
            C=gls_params_tp1.C,
            d=gls_params_tp1.d,
        )
        # assert batch_diag(Vpy).min() > 0
        predictive_distribution = MultivariateNormal(
            loc=mpy_tp1, scale_tril=cholesky(Vpy_tp1),
        )

        log_update = (
            predictive_distribution.log_prob(y)
            + switch_model_dist.log_prob(s_tp1)
            - switch_proposal_dist.log_prob(s_tp1)
        )
        log_weights_tp1 = log_norm_weights + log_update

        return (
            log_weights_tp1,
            s_tp1,
            m_tp1,
            V_tp1,
            mpy_tp1,
            Vpy_tp1,
            gls_params_tp1,
        )

    def _forecast_step(
        self,
        dims,
        s,
        m,
        V,
        seasonal_indicators=None,
        u_state=None,
        u_obs=None,
        u_switch=None,
    ):
        switch_model_dist_tp1 = self._make_switch_model_dist(
            dims=dims, s=s, u_switch=u_switch, m=m, V=V,
        )
        s_tp1 = switch_model_dist_tp1.sample()
        gls_params_tp1 = self.gls_base_parameters(
            switch=s_tp1,
            seasonal_indicators=seasonal_indicators,
            u_state=u_state,
            u_obs=u_obs,
        )
        m_tp1, V_tp1 = filter_forward_prediction_step(
            m=m,
            V=V,
            R=gls_params_tp1.R,
            A=gls_params_tp1.A,
            b=gls_params_tp1.b,
        )
        mpy_tp1, Vpy_tp1 = filter_forward_predictive_distribution(
            m=m_tp1,
            V=V_tp1,
            Q=gls_params_tp1.Q,
            C=gls_params_tp1.C,
            d=gls_params_tp1.d,
        )
        return s_tp1, m_tp1, V_tp1, mpy_tp1, Vpy_tp1, gls_params_tp1

    def _sample_step(
        self,
        dims,
        s,
        x,
        seasonal_indicators,
        u_state=None,
        u_obs=None,
        u_switch=None,
    ):
        switch_model_dist_tp1 = self._make_switch_model_dist(
            dims=dims, s=s, u_switch=u_switch, x=x
        )
        s_tp1 = switch_model_dist_tp1.sample()
        gls_params_tp1 = self.gls_base_parameters(
            switch=s_tp1,
            seasonal_indicators=seasonal_indicators,
            u_state=u_state,
            u_obs=u_obs,
        )

        # covs are not psd in case of ISSM (zeros on most entries).
        # fortunately, these are diagonal -> don't need cholesky, just sqrt of diag.
        if not hasattr(self.gls_base_parameters, "issm"):
            x_tp1 = torch.distributions.MultivariateNormal(
                loc=(
                    matvec(gls_params_tp1.A, x)
                    if gls_params_tp1.A is not None
                    else x
                )
                + (gls_params_tp1.b if gls_params_tp1.b is not None else 0.0),
                scale_tril=cholesky(gls_params_tp1.R),
            ).sample()
        else:
            assert (
                batch_diag_matrix(batch_diag(gls_params_tp1.R))
                == gls_params_tp1.R
            ).all()
            x_tp1 = torch.distributions.MultivariateNormal(
                loc=(
                    matvec(gls_params_tp1.A, x)
                    if gls_params_tp1.A is not None
                    else x
                )
                + (gls_params_tp1.b if gls_params_tp1.b is not None else 0.0),
                scale_tril=batch_diag_matrix(
                    batch_diag(gls_params_tp1.R) ** 0.5
                ),
            ).sample()

        LQ = cholesky(gls_params_tp1.Q)
        y_tp1 = torch.distributions.MultivariateNormal(
            loc=matvec(gls_params_tp1.C, x_tp1)
            + (gls_params_tp1.d if gls_params_tp1.d is not None else 0.0),
            scale_tril=LQ,
        ).sample()
        return s_tp1, x_tp1, y_tp1, gls_params_tp1

    def _make_switch_proposal_dist(
        self,
        switch_model_dist,
        m,
        V,
        y,
        u_state=None,
        u_obs=None,
        u_switch=None,
    ):
        """
        We factorise the proposal distribution as
        \pi(s_t) = p(s_t | s_{t-1}) * q(s_t | y_t) * q(s_t | \psi_t), where
        \psi_t are distribution parameters of p(x_t | y_{1:t-1}. s_{1:t-1}).
        The factor p(s_t | s_{t-1}) is motivated by the form of the
        optimal proposal distribution, and the factorisation of the
        conditional likelihood into two parts allows for dealing
        with missing data (not yet implemented).

        Note: If using RelaxedCategorical (emphasis on relaxed), then
        Product is no longer correct, but approximatively..
        Should not use RelaxedCategorical anyway.
        """
        dists = [switch_model_dist]
        if self.state_to_switch_encoder is not None:
            prev_state_sufficient_statistic = vectorize_normal_dist_params(
                m=m, V=V
            )
            state_to_switch_enc_dist = self.state_to_switch_encoder(
                prev_state_sufficient_statistic
            )
            dists.append(state_to_switch_enc_dist)

        if self.obs_to_switch_encoder is not None:
            concatenated_data = torch.cat(
                tuple(inp for inp in (y, u_switch) if inp is not None), dim=-1
            )
            obs_to_switch_enc_dist = self.obs_to_switch_encoder(
                concatenated_data
            )
            dists.append(obs_to_switch_enc_dist)

        switch_proposal_dist = self.fuse_densities(dists)
        return switch_proposal_dist

    def _make_switch_model_dist(
        self, dims, s, u_switch, x=None, m=None, V=None
    ):
        if s is None:  # initial step --> initial prior
            shp_prepend = (
                (dims.particle, dims.batch)
                if u_switch is None
                else (dims.particle,)
            )
            switch_model_dist = self.switch_prior_model(
                u_switch, batch_shape_to_prepend=shp_prepend,
            )
        else:  # not initial step --> re-sample & transition
            switch_model_dist = self.switch_transition_model(
                u=prepend_batch_dims(u_switch, shp=(dims.particle,))
                if u_switch is not None
                else None,
                s=s,
            )
        return switch_model_dist


class CategoricalSwitchingLinearDynamicalSystem(
    SwitchingLinearDynamicalSystem
):
    def __init__(self, temperature, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._temperature = temperature

    @property
    def temperature(self):
        return self._temperature

    @temperature.setter
    def temperature(self, temperature):
        if isinstance(temperature, float):
            temperature = torch.tensor(temperature)
        self._temperature = temperature

    def _make_switch_model_dist(self, *args, **kwargs):
        if self.temperature is None:
            return super()._make_switch_model_dist(*args, **kwargs)
        else:
            return StableRelaxedOneHotCategorical(
                logits=super()._make_switch_model_dist(*args, **kwargs).logits,
                temperature=self.temperature,
            )

    def _make_switch_proposal_dist(self, *args, **kwargs):
        if self.temperature is None:
            return super()._make_switch_proposal_dist(*args, **kwargs)
        else:
            return StableRelaxedOneHotCategorical(
                logits=super()
                ._make_switch_proposal_dist(*args, **kwargs)
                .logits,
                temperature=self.temperature,
            )


class RecurrentSwitchingLinearDynamicalSystem(SwitchingLinearDynamicalSystem):
    def _make_switch_model_dist(
        self, dims, s, u_switch, x=None, m=None, V=None
    ):
        assert len({(x is None), (m is None and V is None)}) == 2

        if s is None:  # initial step --> initial prior
            shp_prepend = (
                (dims.particle, dims.batch)
                if u_switch is None
                else (dims.particle,)
            )
            switch_model_dist = self.switch_prior_model(
                u_switch, batch_shape_to_prepend=shp_prepend,
            )
        else:  # not initial step --> re-sample & transition
            switch_model_dist = self.switch_transition_model(
                u=prepend_batch_dims(u_switch, shp=(dims.particle,))
                if u_switch is not None
                else None,
                s=s,
                x=x,
                m=m,
                V=V,
            )
        return switch_model_dist


class SGLSPredictor(RepresentablePredictor):
    """ wrapper to to allow make_evaluation_predictions to evaluate this model. """

    def __init__(
        self,
        model: SwitchingLinearDynamicalSystem,
        input_transform,
        batch_size: int,
        prediction_length: int,
        freq: str,
        cardinalities,
        dims,
        bias_y,
        factor_y,
        time_feat,
        lead_time: int = 0,
        keep_filtered_predictions=True,
        yield_forecast_only=False,
    ):
        super().__init__(
            prediction_length=prediction_length, freq=freq, lead_time=lead_time
        )
        self.model = model.module if hasattr(model, "module") else model
        self.parallel_model_fn = model
        self.input_transform = input_transform
        self.batch_size = batch_size
        self.cardinalities = cardinalities
        self.dims = dims
        self.bias_y = bias_y
        self.factor_y = factor_y
        self.time_feat = time_feat
        # gluonts calls predict without alllowed kwargs from outside in backtest.py...
        # therefore we set this option via mutable attributes. sweet.
        self.keep_filtered_predictions = keep_filtered_predictions
        self.yield_forecast_only = yield_forecast_only

    def predict(
        self,
        dataset: Dataset,
        num_samples,
        num_workers: Optional[int] = None,
        num_prefetch: Optional[int] = None,
        **kwargs,
    ) -> Iterator[Dict[Forecast, torch.Tensor]]:
        assert (
            num_samples == self.model.n_particle
        ), "set n_particle from outside"
        inference_loader = InferenceDataLoader(
            dataset=dataset,
            transform=self.input_transform,
            batch_size=self.batch_size,
            num_workers=num_workers,
            num_prefetch=num_prefetch,
            ctx=None,
            dtype=np.float32,
            **kwargs,
        )

        for batch_gts in inference_loader:
            batch = transform_gluonts_to_pytorch(
                batch=batch_gts,
                bias_y=self.bias_y,
                factor_y=self.factor_y,
                device=self.model.state_prior_model.m.device,
                dtype=self.model.state_prior_model.m.dtype,
                time_features=self.time_feat,
                **self.cardinalities,
            )

            (
                losses_filter_time_batch_wise,
                y_forecast,
                mpy_filter,
                Vpy_filter,
            ) = self.parallel_model_fn(
                **batch,
                make_forecast=True,
                n_steps_forecast=self.prediction_length,
            )

            # TODO: bad style to re-assign same variables.
            #  Also hide this in a method.
            # compute losses  # TB -> BT (P already marginalised)
            losses_gts = losses_filter_time_batch_wise.transpose(0, 1)
            losses_gts = losses_gts.detach().cpu().numpy()

            # scale filter predictive accordingly and transform to gts format
            mpy_gts = torch.stack(mpy_filter, dim=0)
            mpy_gts = (mpy_gts * self.factor_y) + self.bias_y
            mpy_gts = mpy_gts.transpose(0, 2)  # TPBF -> BPTF
            mpy_gts = mpy_gts.detach().cpu().numpy()
            Vpy_gts = torch.stack(Vpy_filter, dim=0)
            Vpy_gts = Vpy_gts * (self.factor_y ** 2)
            Vpy_gts = Vpy_gts.transpose(0, 2)  # TPBF -> BPTF
            Vpy_gts = Vpy_gts.detach().cpu().numpy()

            # transform forecast into GTS format
            forecast_gts = torch.stack(y_forecast, dim=0)
            forecast_gts = (forecast_gts * self.factor_y) + self.bias_y
            forecast_gts = forecast_gts.transpose(0, 2)  # TPBF -> BPTF
            forecast_gts = forecast_gts.detach().cpu().numpy()
            forecast_gts = forecast_gts.squeeze(
                axis=-1
            )  # this is bad, but backtest requires it.

            # return iterator over both forecast object and filter objects.
            for (
                idx_sample_in_batch,
                (_fcst_gts, _loss_gts, _mpy_gts, _Vpy_gts),
            ) in enumerate(zip(forecast_gts, losses_gts, mpy_gts, Vpy_gts)):
                if self.yield_forecast_only:
                    yield SampleForecast(
                        samples=_fcst_gts,
                        start_date=batch_gts["forecast_start"][
                            idx_sample_in_batch
                        ],
                        freq=self.freq,
                        item_id=batch[FieldName.ITEM_ID][idx_sample_in_batch]
                        if FieldName.ITEM_ID in batch
                        else None,
                        info=batch["info"][idx_sample_in_batch]
                        if "info" in batch
                        else None,
                    )
                else:
                    yield {
                        "forecast": SampleForecast(
                            samples=_fcst_gts,
                            start_date=batch_gts["forecast_start"][
                                idx_sample_in_batch
                            ],
                            freq=self.freq,
                            item_id=batch[FieldName.ITEM_ID][
                                idx_sample_in_batch
                            ]
                            if FieldName.ITEM_ID in batch
                            else None,
                            info=batch["info"][idx_sample_in_batch]
                            if "info" in batch
                            else None,
                        ),
                        "loss": _loss_gts,
                        "mpy_filter": _mpy_gts,
                        "Vpy_filter": _Vpy_gts,
                    }
            assert idx_sample_in_batch + 1 == len(batch_gts["forecast_start"])
