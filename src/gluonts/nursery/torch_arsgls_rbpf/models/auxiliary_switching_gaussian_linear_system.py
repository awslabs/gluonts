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
    make_argmax_log_weights,
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
from torch_extensions.fusion import ProbabilisticSensorFusion
from torch_extensions.distributions.parametrised_distribution import (
    ParametrisedDistribution,
    ParametrisedMultivariateNormal,
    prepend_batch_dims,
)
from models.gls_parameters import GLSParameters
from models.dynamical_system import DynamicalSystem
from experiments.model_component_zoo.input_transforms import InputTransformer
from experiments.model_component_zoo.input_transforms import ControlInputs
from data.gluonts_nips_datasets.gluonts_nips_datasets import (
    transform_gluonts_to_pytorch,
)


# TODO: share methods with SwitchingLinearDynamicalSystem!
#  Not clear how to share with Recurrent. Double inheritance / Maybe Mixin?
class AuxiliarySwitchingLinearDynamicalSystem(DynamicalSystem):
    def __init__(
        self,
        n_state: int,
        n_obs: int,
        n_ctrl_state: int,
        n_particle: int,
        n_switch: int,
        gls_base_parameters: GLSParameters,
        input_transformer: (InputTransformer, None),
        measurement_model: nn.Module,
        obs_encoder: nn.Module,
        state_prior_model: ParametrisedMultivariateNormal,
        switch_prior_model: ParametrisedDistribution,
        switch_transition_model: nn.Module,
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
        self.switch_transition_model = switch_transition_model
        self.measurement_model = measurement_model
        self.obs_encoder = obs_encoder
        self.fuse_densities = ProbabilisticSensorFusion()
        self.state_prior_model = state_prior_model
        self.switch_prior_model = switch_prior_model
        self.resampling_criterion_fn = make_criterion_fn_with_ess_threshold(
            min_ess_ratio=min_ess_ratio
        )
        self.resampling_indices_fn = resampling_indices_fn

        # TODO: same in KVAE. rework this. maybe share such optional model structure in a base class
        if input_transformer is None:

            def none_inputs(u_static_cat=None, u_time=None):
                assert (
                    u_static_cat == None
                ), "controls were given, however, input transform is None"
                assert (
                    u_time == None
                ), "controls were given, however, input transform is None"
                return ControlInputs(state=None, obs=None, switch=None)

            self.input_transformer = none_inputs
        else:
            self.input_transformer = input_transformer

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
        make_forecast=False,
        n_steps_forecast=None,
    ):
        # TODO: see corr. TODO in switching_gaussian_linear_system.py

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
            )
            (
                log_weights_filter,
                s_filter,
                z_filter,
                m_filter,
                V_filter,
                gls_params_filter,
            ) = filter_results
            y_filter_dist = self.measurement_model(
                torch.stack(z_filter, dim=0)
            )

            # Resample
            # We don't need z - it has no temporal dependencies. We sample it s -> x -> z -> y.
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
            x = MultivariateNormal(loc=m, covariance_matrix=V).rsample()

            # Sample from generative model forward
            (
                s_forecast,
                x_forecast,
                z_forecast,
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
            y_forecast_dist = self.measurement_model(
                torch.stack(z_forecast, dim=0)
            )
            log_conditional_likelihoods_batchwise = tuple(
                torch.logsumexp(lws, dim=0) for lws in log_weights_filter
            )
            log_marginal_likelihood = torch.stack(
                log_conditional_likelihoods_batchwise, dim=0
            )
            losses_time_batch_wise = -log_marginal_likelihood
            return losses_time_batch_wise, y_forecast_dist, y_filter_dist
        else:
            return self.loss_forward(
                y=y,
                u_static_cat=u_static_cat,
                u_time=u_time,
                seasonal_indicators=seasonal_indicators,
            )

    def loss_forward(
        self, y, u_static_cat=None, u_time=None, seasonal_indicators=None
    ):
        """ computes loss TB-wise """
        log_weights, s, z, m, V, gls_params = self.filter_forward(
            y=y,
            u_static_cat=u_static_cat,
            u_time=u_time,
            seasonal_indicators=seasonal_indicators,
        )
        # sum over batches and  LSE over particle dim.
        log_conditional_likelihoods_batchwise = torch.stack(
            [torch.logsumexp(lws, dim=0) for lws in log_weights], dim=0
        )
        return -log_conditional_likelihoods_batchwise

    def filter_forward(
        self, y, u_static_cat=None, u_time=None, seasonal_indicators=None
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
        log_weights, m, V, s, z, gls_params = (
            [None] * dims.timesteps for _ in range(6)
        )

        for t in range(dims.timesteps):
            if t == 0:
                state_prior = self.state_prior_model(
                    None, batch_shape_to_prepend=(dims.particle, dims.batch)
                )
                (
                    log_weights[t],
                    s[t],
                    z[t],
                    m[t],
                    V[t],
                    gls_params[t],
                ) = self._filter_forward_step(
                    dims=dims,
                    log_weights=create_zeros_log_weights(
                        dims=dims, device=device, dtype=dtype
                    )[0],
                    s=None,
                    z=None,
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
                    z[t],
                    m[t],
                    V[t],
                    gls_params[t],
                ) = self._filter_forward_step(
                    dims=dims,
                    log_weights=log_weights[t - 1],
                    s=s[t - 1],
                    z=z[t - 1],
                    m=m[t - 1],
                    V=V[t - 1],
                    y=y[t],
                    u_state=u_state[t],
                    u_obs=u_obs[t],
                    u_switch=u_switch[t],
                    seasonal_indicators=seasonal_indicators[t],
                )
        gls_params = list_of_dicts_to_dict_of_list(gls_params)
        return log_weights, s, z, m, V, gls_params

    def sample(
        self,
        s=None,
        x=None,
        u_static_cat=None,
        u_time=None,
        seasonal_indicators=None,
        n_timesteps=None,
        n_batch=None,
        deterministic=False,
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
            x_initial = (
                state_prior.mean if deterministic else state_prior.rsample()
            )
            s_initial = None

        (u_state, u_obs, u_switch, seasonal_indicators) = (
            [None] * dims.timesteps if u is None else u
            for u in (u_state, u_obs, u_switch, seasonal_indicators)
        )
        (s, x, z, gls_params) = ([None] * dims.timesteps for _ in range(4))

        for t in range(n_timesteps):
            s[t], x[t], z[t], gls_params[t] = self._sample_step(
                dims=dims,
                s=s[t - 1] if t > 0 else s_initial,
                x=x[t - 1] if t > 0 else x_initial,
                seasonal_indicators=seasonal_indicators[t],
                u_state=u_state[t],
                u_obs=u_obs[t],
                u_switch=u_switch[t],
                deterministic=deterministic,
            )
        return s, x, z, list_of_dicts_to_dict_of_list(gls_params)

    def filter_forecast(
        self,
        n_steps_forecast,
        y,
        u_time=None,
        u_static_cat=None,
        seasonal_indicators=None,
        deterministic=False,
    ):
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
            z_filter,
            m_filter,
            V_filter,
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
        log_weights, s, z, m, V = (
            log_weights_filter[-1],
            s_filter[-1],
            z_filter[-1],
            m_filter[-1],
            V_filter[-1],
        )

        log_norm_weights, (s, z, m, V) = resample(
            n_particle=dims.particle,
            log_norm_weights=normalize_log_weights(
                log_weights=log_weights
                if not deterministic
                else make_argmax_log_weights(log_weights),
            ),
            tensors_to_resample=(s, z, m, V),
            resampling_indices_fn=self.resampling_indices_fn,
            criterion_fn=make_criterion_fn_with_ess_threshold(
                min_ess_ratio=1.0
            ),  # always
        )
        # Forecast
        x = (
            MultivariateNormal(loc=m, covariance_matrix=V).rsample()
            if not deterministic
            else m
        )

        # Sample from generative model forward
        s_forecast, x_forecast, z_forecast, gls_params_forecast = self.sample(
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
            deterministic=deterministic,
        )
        # Concatenate filter and forecast
        assert all(
            [
                isinstance(item, (list, tuple, set))
                for item in [s_filter, s_forecast, m_filter, V_filter]
            ]
        )
        s_trajectory = torch.stack(tuple(s_filter) + tuple(s_forecast), dim=0)
        z_trajectory = torch.stack(tuple(z_filter) + tuple(z_forecast), dim=0)
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

        y_trajectory_dist = self.measurement_model(z_trajectory)

        return (
            log_norm_weights_trajectory,
            s_trajectory,
            z_trajectory,
            y_trajectory_dist,
            gls_params_trajectory,
        )

    def _filter_forward_step(
        self,
        dims,
        log_weights,
        s,
        z,
        m,
        V,
        y,
        u_state=None,
        u_obs=None,
        u_switch=None,
        seasonal_indicators=None,
    ):
        log_norm_weights = normalize_log_weights(log_weights=log_weights)
        assert len({s is None, z is None}) == 1
        if s is not None:
            log_norm_weights, (s, z, m, V) = resample(
                n_particle=dims.particle,
                log_norm_weights=log_norm_weights,
                tensors_to_resample=(s, z, m, V),
                resampling_indices_fn=self.resampling_indices_fn,
                criterion_fn=self.resampling_criterion_fn,
            )
        switch_model_dist = self._make_switch_model_dist(
            dims=dims, s=s, u_switch=u_switch, m=m, V=V,
        )

        # TODO: u_switch is not correct for both s and z.
        #  But at the moment there is only one u_* anyway (copies).
        #  Needs redesign of inputs.
        concatenated_data = torch.cat(
            tuple(inp for inp in (y, u_switch) if inp is not None), dim=-1
        )
        encoder_dists = self.obs_encoder(concatenated_data)

        switch_proposal_dist = self._make_switch_proposal_dist(
            switch_model_dist=switch_model_dist,
            switch_encoder_dist=encoder_dists.switch,
        )
        s_tp1 = switch_proposal_dist.rsample()

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

        auxiliary_model_dist = self._make_auxiliary_model_dist(
            mp=mp, Vp=Vp, gls_params=gls_params_tp1
        )
        auxiliary_proposal_dist = self._make_auxiliary_proposal_dist(
            auxiliary_model_dist=auxiliary_model_dist,
            auxiliary_encoder_dist=encoder_dists.auxiliary,
        )
        z_tp1 = auxiliary_proposal_dist.rsample()

        m_tp1, V_tp1 = filter_forward_measurement_step(
            y=z_tp1,
            m=mp,
            V=Vp,
            Q=gls_params_tp1.Q,
            C=gls_params_tp1.C,
            d=gls_params_tp1.d,
        )

        measurement_dist = self.measurement_model(z_tp1)

        log_update = (
            measurement_dist.log_prob(y)
            + auxiliary_model_dist.log_prob(z_tp1)
            + switch_model_dist.log_prob(s_tp1)
            - switch_proposal_dist.log_prob(s_tp1)
            - auxiliary_proposal_dist.log_prob(z_tp1)
        )
        log_weights_tp1 = log_norm_weights + log_update

        return log_weights_tp1, s_tp1, z_tp1, m_tp1, V_tp1, gls_params_tp1

    def _sample_step(
        self,
        dims,
        s,
        x,
        seasonal_indicators,
        u_state=None,
        u_obs=None,
        u_switch=None,
        deterministic=False,
    ):
        switch_model_dist_tp1 = self._make_switch_model_dist(
            dims=dims, s=s, u_switch=u_switch, x=x
        )
        s_tp1 = (
            switch_model_dist_tp1.mean
            if deterministic
            else switch_model_dist_tp1.rsample()
        )
        gls_params_tp1 = self.gls_base_parameters(
            switch=s_tp1,
            seasonal_indicators=seasonal_indicators,
            u_state=u_state,
            u_obs=u_obs,
        )

        # covs are not psd in case of ISSM (zeros on most entries).
        # fortunately, these are diagonal -> don't need cholesky, just sqrt of diag.
        try:
            x_tp1_dist = torch.distributions.MultivariateNormal(
                loc=(
                    matvec(gls_params_tp1.A, x)
                    if gls_params_tp1.A is not None
                    else x
                )
                + (gls_params_tp1.b if gls_params_tp1.b is not None else 0.0),
                covariance_matrix=gls_params_tp1.R,
            )
        except:
            assert (
                batch_diag_matrix(batch_diag(gls_params_tp1.R))
                == gls_params_tp1.R
            ).all()
            x_tp1_dist = torch.distributions.MultivariateNormal(
                loc=(
                    matvec(gls_params_tp1.A, x)
                    if gls_params_tp1.A is not None
                    else x
                )
                + (gls_params_tp1.b if gls_params_tp1.b is not None else 0.0),
                scale_tril=batch_diag_matrix(
                    batch_diag(gls_params_tp1.R) ** 0.5
                ),
            )

        x_tp1 = x_tp1_dist.mean if deterministic else x_tp1_dist.rsample()

        LQ = cholesky(gls_params_tp1.Q)
        z_tp1_dist = torch.distributions.MultivariateNormal(
            loc=matvec(gls_params_tp1.C, x_tp1)
            + (gls_params_tp1.d if gls_params_tp1.d is not None else 0.0),
            scale_tril=LQ,
        )
        z_tp1 = z_tp1_dist.mean if deterministic else z_tp1_dist.rsample()

        return s_tp1, x_tp1, z_tp1, gls_params_tp1

    def _make_switch_proposal_dist(
        self, switch_model_dist, switch_encoder_dist
    ):
        switch_proposal_dist = self.fuse_densities(
            [switch_model_dist, switch_encoder_dist]
        )
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

    def _make_auxiliary_proposal_dist(
        self, auxiliary_model_dist, auxiliary_encoder_dist
    ):
        return self.fuse_densities(
            [auxiliary_model_dist, auxiliary_encoder_dist]
        )

    def _make_auxiliary_model_dist(self, mp, Vp, gls_params):
        mpz_tp1, Vpz_tp1 = filter_forward_predictive_distribution(
            m=mp, V=Vp, Q=gls_params.Q, C=gls_params.C, d=gls_params.d
        )
        auxiliary_model_dist = MultivariateNormal(
            loc=mpz_tp1, scale_tril=cholesky(Vpz_tp1)
        )
        return auxiliary_model_dist


class RecurrentAuxiliarySwitchingLinearDynamicalSystem(
    AuxiliarySwitchingLinearDynamicalSystem
):
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
        model: AuxiliarySwitchingLinearDynamicalSystem,
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
                y_forecast_dist,
                y_filter_dist,
            ) = self.parallel_model_fn(
                **batch,
                make_forecast=True,
                n_steps_forecast=self.prediction_length,
            )

            # compute losses  # TB -> BT (P already marginalised)
            losses_gts = losses_filter_time_batch_wise.transpose(0, 1)
            losses_gts = losses_gts.detach().cpu().numpy()

            # transform forecast into GTS format
            forecast_gts = y_forecast_dist.rsample()
            forecast_gts = (forecast_gts * self.factor_y) + self.bias_y
            forecast_gts = forecast_gts.transpose(0, 2)  # TPBF -> BPTF
            forecast_gts = forecast_gts.detach().cpu().numpy()
            forecast_gts = forecast_gts.squeeze(
                axis=-1
            )  # this is bad, but backtest requires it.

            y_filter_gts = y_filter_dist.rsample()
            y_filter_gts = (y_filter_gts * self.factor_y) + self.bias_y
            y_filter_gts = y_filter_gts.transpose(0, 2)  # TPBF -> BPTF
            y_filter_gts = y_filter_gts.detach().cpu().numpy()
            y_filter_gts = y_filter_gts.squeeze(axis=-1)

            # return iterator over both forecast object and filter objects.
            for (
                idx_sample_in_batch,
                (_fcst_gts, _loss_gts, _y_filter_gts),
            ) in enumerate(zip(forecast_gts, losses_gts, y_filter_gts)):
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
                        "y_filter": _y_filter_gts,
                    }
            assert idx_sample_in_batch + 1 == len(batch_gts["forecast_start"])
