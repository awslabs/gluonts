from box import Box
import numpy as np
import torch
from torch import nn
from torch.distributions import MultivariateNormal
from utils.utils import make_inv_tril_parametrization, TensorDims
from torch_extensions.ops import matvec
from inference.analytical_gausian_linear.inference_sequence_inhomogenous import (
    filter_forward,
    smooth_forward_backward,
    loss_em,
)
from inference.analytical_gausian_linear.inference_step import (
    filter_forward_predictive_distribution,
)
from models.dynamical_system import DynamicalSystem
from models.gls_parameters.gls_parameters import GLSParameters
from torch_extensions.distributions.parametrised_distribution import (
    ParametrisedMultivariateNormal,
)
from models_new_will_replace.dynamical_system import ControlInputs


# TODO: Add missing data code from the diverged branch.

# TODO: introduced the same shitty way of having 2 types of ctrls
#  u_time, u_static -> u_obs, u_state. Re-design!!!
#  Either make it more closely to gluon-TS or remove gluonTS dependency...
class KalmanVariationalAutoEncoder(DynamicalSystem):
    def __init__(
        self,
        n_state: int,
        n_obs: int,
        n_auxiliary,
        n_ctrl_state: int,
        n_particle: int,
        gls_base_parameters: GLSParameters,
        measurement_model: nn.Module,
        obs_to_auxiliary_encoder: nn.Module,
        rnn_switch_model: nn.RNNBase,
        state_prior_model: ParametrisedMultivariateNormal,
        input_transformer: (nn.Module, None) = None,
        reconstruction_weight: float = 1.0,
    ):
        super().__init__(
            n_state=n_state,
            n_obs=n_obs,
            n_ctrl_state=n_ctrl_state,
            n_particle=n_particle,
        )
        self.n_auxiliary = n_auxiliary
        self.gls_base_parameters = gls_base_parameters
        self.measurement_model = measurement_model
        self.obs_to_auxiliary_encoder = obs_to_auxiliary_encoder
        self.rnn_switch_model = rnn_switch_model
        self.state_prior_model = state_prior_model
        self.z_initial = torch.nn.Parameter(torch.zeros(self.n_auxiliary))
        self.reconstruction_weight = reconstruction_weight

        # TODO: name too close to "input_transform" for the predictor object.
        #  However, all this is anyways a byproduct of some gluonTS dependency.
        #  needs to be redesigned!
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
        dims.update(auxiliary=self.n_auxiliary)
        dims = TensorDims(**dims)
        return dims

    def expand_particle_dim(self, u: ControlInputs):
        # assumes we have time dimension
        u.target = u.target[:, None, ...] if u.target is not None else None
        u.state = u.state[:, None, ...] if u.state is not None else None
        u.switch = u.switch[:, None, ...] if u.switch is not None else None
        return u

    def modify_obs_dims_for_ssm(self, dims):
        dims = dims._asdict()
        dims.update(obs=self.n_auxiliary)
        dims = TensorDims(**dims)
        return dims

    def loss_em(
        self, y, u_static_cat=None, u_time=None, rao_blackwellized=True
    ):
        if rao_blackwellized:
            return self._loss_em_rb(
                y=y, u_static_cat=u_static_cat, u_time=u_time
            )
        else:
            return self._loss_em_mc(
                y=y, u_static_cat=u_static_cat, u_time=u_time
            )

    def _loss_em_rb(self, y, u_static_cat=None, u_time=None):
        """ Rao-Blackwellization for part of the loss (the EM loss term of the SSM). """
        u = self.expand_particle_dim(
            self.input_transformer(u_static_cat=u_static_cat, u_time=u_time)
        )
        dims = self.get_dims(y=y)

        q = self.obs_to_auxiliary_encoder(y)
        # SSM pseudo observations. For log_prob evaluation, need particle_first, for RNN time_first.
        z_particle_first = q.rsample([dims.particle])
        z_time_first = z_particle_first.transpose(0, 1)

        z_initial = self.z_initial[None, None, None, :].repeat(
            1, dims.particle, dims.batch, 1
        )
        rnn_inputs = torch.cat([z_initial, z_time_first[:-1]], dim=0)

        gls_params, rnn_state = self.compute_gls_params(
            rnn_inputs=rnn_inputs, u_state=u.state, u_obs=u.target,
        )
        LQinv_tril, LQinv_logdiag = make_inv_tril_parametrization(gls_params.Q)
        LRinv_tril, LRinv_logdiag = make_inv_tril_parametrization(gls_params.R)

        state_prior = self.state_prior_model(
            None, batch_shape_to_prepend=(dims.particle, dims.batch)
        )
        LV0inv_tril, LV0inv_logdiag = make_inv_tril_parametrization(
            state_prior.covariance_matrix
        )
        m0 = state_prior.loc

        l_em = (
            loss_em(
                dims=self.modify_obs_dims_for_ssm(dims=dims),
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
                u_state=u.state,
                u_obs=u.target,
            ).sum(dim=0)
            / dims.particle
        )  # loss_em fn already sums over time. Only avg Particle dim.
        l_measurement = (
            -self.measurement_model(z_particle_first)
            .log_prob(y)
            .sum(dim=(0, 1))
            / dims.particle
        )  # Time and Particle
        l_auxiliary_encoder = (
            q.log_prob(z_particle_first).sum(dim=(0, 1)) / dims.particle
        )  # Time and Particle
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

    def _loss_em_mc(self, y, u_static_cat=None, u_time=None):
        """" Monte Carlo only loss as computed in KVAE paper """
        u = self.expand_particle_dim(
            self.input_transformer(u_static_cat=u_static_cat, u_time=u_time)
        )
        dims = self.get_dims(y=y)

        # A) SSM related distributions:
        # A1) smoothing.
        # TODO: This re-computes the input transformation inside smoothing.
        #  This needs re-design. However, KVAE usually has no input transform.
        #  We just need the optional embedding, otherwise it would perform bad.
        (
            x_smoothed_dist,
            z_filter_dist,
            gls_params,
            last_rnn_state,
            inv_measurement_dist,
            z,
        ) = self.smooth_forward_backward(
            y=y, u_static_cat=u_static_cat, u_time=u_time
        )
        x = x_smoothed_dist.rsample()

        # A2) prior && posterior transition distribution.
        prior_dist = self.state_prior_model(
            None, batch_shape_to_prepend=(dims.particle, dims.batch)
        )
        #  # A, B, R are already 0:T-1.

        transition_dist = MultivariateNormal(
            loc=matvec(gls_params.A[:-1], x[:-1])
            + (
                matvec(gls_params.B[:-1], u.state[:-1])
                if gls_params.B is not None
                else 0.0
            ),
            covariance_matrix=gls_params.R[:-1],
        )
        # A3) posterior predictive (auxiliary) distribution.
        auxiliary_predictive_dist = MultivariateNormal(
            loc=matvec(gls_params.C, x)
            + (
                matvec(gls_params.D, u.target)
                if gls_params.D is not None
                else 0.0
            ),
            covariance_matrix=gls_params.Q,
        )

        # A4) SSM related losses
        l_prior = (
            -prior_dist.log_prob(x[0:1]).sum(dim=(0, 1)) / dims.particle
        )  # time and particle dim
        l_transition = (
            -transition_dist.log_prob(x[1:]).sum(dim=(0, 1)) / dims.particle
        )  # time and particle dim
        l_auxiliary = (
            -auxiliary_predictive_dist.log_prob(z).sum(dim=(0, 1))
            / dims.particle
        )  # time and particle dim
        l_entropy = (
            x_smoothed_dist.log_prob(x).sum(dim=(0, 1))  # negative entropy
            / dims.particle
        )  # time and particle dim

        # B) VAE related distributions
        # B1) inv_measurement_dist already obtained from smoothing (as we dont want to re-compute)
        # B2) measurement (decoder) distribution
        # transpose TPBF -> PTBF to broadcast log_prob of y (TBF) correctly
        z_particle_first = z.transpose(0, 1)
        measurement_dist = self.measurement_model(z_particle_first)
        # B3) VAE related losses
        l_measurement = (
            -measurement_dist.log_prob(y).sum(dim=(0, 1)) / dims.particle
        )  # time and particle dim
        l_inv_measurement = (
            inv_measurement_dist.log_prob(z_particle_first).sum(dim=(0, 1))
            / dims.particle
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

    def filter_forward(self, y, u_static_cat=None, u_time=None):
        u = self.expand_particle_dim(
            self.input_transformer(u_static_cat=u_static_cat, u_time=u_time)
        )
        dims = self.get_dims(y=y)

        state_prior = self.state_prior_model(
            None, batch_shape_to_prepend=(dims.particle, dims.batch)
        )
        LV0inv_tril, LV0inv_logdiag = make_inv_tril_parametrization(
            state_prior.covariance_matrix
        )
        m0 = state_prior.loc

        # Encode observations y[0:T] to obtain all pseudo-observation z[0:T]
        inv_measurement_dist = self.obs_to_auxiliary_encoder(y)
        z = inv_measurement_dist.rsample([dims.particle]).transpose(0, 1)
        # Use as RNN input [z_initial, z[0:T-1]], i.e. previous pseudo-observation.
        z_initial = self.z_initial[None, None, None, :].repeat(
            1, dims.particle, dims.batch, 1
        )
        rnn_inputs = torch.cat([z_initial, z[:-1]], dim=0)

        # Unroll RNN on all pseudo-obervations to get the SSM params
        gls_params, last_rnn_state = self.compute_gls_params(
            rnn_inputs=rnn_inputs, u_state=u.state, u_obs=u.target,
        )

        # filter with pseudo-obs.
        # TODO: Filter is currently only implemented with
        #  cholesky parametrization. Overload that function (but its python...)?
        LQinv_tril, LQinv_logdiag = make_inv_tril_parametrization(gls_params.Q)
        LRinv_tril, LRinv_logdiag = make_inv_tril_parametrization(gls_params.R)
        m_fw, V_fw = filter_forward(
            dims=self.modify_obs_dims_for_ssm(dims),
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
            u_state=u.state,
            u_obs=u.target,
        )
        # predictive distribution of SSM is VAE latent z.
        mpz, Vpz = filter_forward_predictive_distribution(
            m=m_fw,
            V=V_fw,
            Q=gls_params.Q,
            C=gls_params.C,
            d=matvec(gls_params, u.target) if gls_params.D is not None else None,
        )

        x_filter_dist = MultivariateNormal(loc=m_fw, covariance_matrix=V_fw)
        z_filter_dist = MultivariateNormal(loc=mpz, covariance_matrix=Vpz)

        return (
            x_filter_dist,
            z_filter_dist,
            gls_params,
            last_rnn_state,
            inv_measurement_dist,
            z,
        )

    def smooth_forward_backward(self, y, u_static_cat=None, u_time=None):
        # TODO: most of the stuff is duplicated code compared to filter_forward. put in methods

        u = self.expand_particle_dim(
            self.input_transformer(u_static_cat=u_static_cat, u_time=u_time)
        )
        dims = self.get_dims(y=y)

        state_prior = self.state_prior_model(
            None, batch_shape_to_prepend=(dims.particle, dims.batch)
        )
        LV0inv_tril, LV0inv_logdiag = make_inv_tril_parametrization(
            state_prior.covariance_matrix
        )
        m0 = state_prior.loc

        inv_measurement_dist = self.obs_to_auxiliary_encoder(y)
        z = inv_measurement_dist.rsample([dims.particle]).transpose(0, 1)

        z_initial = self.z_initial[None, None, None, :].repeat(
            1, dims.particle, dims.batch, 1
        )
        rnn_inputs = torch.cat([z_initial, z[:-1]], dim=0)

        gls_params, last_rnn_state = self.compute_gls_params(
            rnn_inputs=rnn_inputs, u_state=u.state, u_obs=u.target
        )
        LQinv_tril, LQinv_logdiag = make_inv_tril_parametrization(gls_params.Q)
        LRinv_tril, LRinv_logdiag = make_inv_tril_parametrization(gls_params.R)

        m_fb, V_fb, Cov_fb = smooth_forward_backward(
            dims=self.modify_obs_dims_for_ssm(dims),
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
            u_state=u.state,
            u_obs=u.target,
        )
        # predictive distribution of SSM is VAE latent z.
        mpz, Vpz = filter_forward_predictive_distribution(
            m=m_fb,
            V=V_fb,
            Q=gls_params.Q,
            C=gls_params.C,
            d=matvec(gls_params, u.target) if gls_params.D is not None else None,
        )
        # We omit the covariance between states. Could add it if we need.
        x_smoothed_dist = MultivariateNormal(loc=m_fb, covariance_matrix=V_fb)
        z_filter_dist = MultivariateNormal(loc=mpz, covariance_matrix=Vpz)

        return (
            x_smoothed_dist,
            z_filter_dist,
            gls_params,
            last_rnn_state,
            inv_measurement_dist,
            z,
        )

    def sample(
        self,
        x,
        z,
        gls_params,
        rnn_state,
        u_static_cat=None,
        u_time=None,
        n_timesteps=None,
        n_batch=None,
        deterministic=False,
    ):
        # all inputs are assumed from time-step t-1, in which case this is forecasting.
        # or None, in which case we sample from prior entirely generatively.
        u = self.expand_particle_dim(
            self.input_transformer(u_static_cat=u_static_cat, u_time=u_time)
        )
        dims = self.get_dims(
            u_state=u.state,
            u_obs=u.target,
            n_timesteps=n_timesteps,
            n_batch=n_batch,
        )

        assert (
            len(
                set(tensor is None for tensor in (x, z, rnn_state, gls_params))
            )
            == 1
        )
        sample_from_prior = all(
            tensor is None for tensor in (x, z, rnn_state, gls_params)
        )

        if sample_from_prior:
            raise NotImplementedError("no time for this.")
        else:
            x_initial = x
            z_initial = z
            gls_params_initial = gls_params
            rnn_state_initial = rnn_state

        (x, z, rnn_state, gls_params) = (
            [None] * dims.timesteps for _ in range(4)
        )
        for t in range(n_timesteps):
            x[t], z[t], gls_params[t], rnn_state[t] = self._sample_step(
                x_tm1=x[t - 1] if t > 0 else x_initial,
                z_tm1=z[t - 1] if t > 0 else z_initial,
                gls_params_tm1=gls_params[t - 1]
                if t > 0
                else gls_params_initial,
                rnn_state_tm1=rnn_state[t - 1] if t > 0 else rnn_state_initial,
                u_state_t=u.state[t] if u.state is not None else None,
                u_obs_t=u.target[t] if u.target is not None else None,
                deterministic=deterministic,
            )
        x = torch.stack(x, dim=0)
        z = torch.stack(z, dim=0)

        gls_params = Box(
            {
                name: [param[name] for param in gls_params]
                for name in gls_params[0].keys()
            }
        )
        gls_params = Box(
            {
                name: torch.stack(params, dim=0)
                if not all(p is None for p in params)
                else None
                for name, params in gls_params.items()
            }
        )
        return x, z, gls_params, rnn_state

    def _sample_step(
        self,
        x_tm1,
        z_tm1,
        gls_params_tm1,
        rnn_state_tm1,
        u_state_t,
        u_obs_t,
        deterministic=False,
    ):
        # assumes inputs do not have time-dim. except rnn_state
        # must add time-dim to rnn inputs and remove it from outputs.
        # TODO: make this consistent. either remove from rnn_state as well or have it everywhere.
        rnn_inputs = z_tm1[None, ...]  # add time-dim of 1
        shp = z_tm1.shape  # do not include t-dim
        shp_flat = (1,) + (np.prod(shp[:-1]),) + (shp[-1],)
        rnn_output_t_, rnn_state_t = self.rnn_switch_model(
            rnn_inputs.reshape(shp_flat),
            rnn_state_tm1,  # already has t-dim of 1.
        )
        rnn_output_t = rnn_output_t_[0]  # remove the t-dim.

        rnn_output_t = rnn_output_t.reshape(
            shp[:-1] + (rnn_output_t.shape[-1],)
        )
        gls_params_t = self.gls_base_parameters(
            switch=rnn_output_t, u_state=u_state_t, u_obs=u_obs_t,
        )
        x_t_dist = torch.distributions.MultivariateNormal(
            loc=(
                matvec(gls_params_tm1.A, x_tm1)
                if gls_params_tm1.A is not None
                else x_tm1
            )
            + (gls_params_tm1.b if gls_params_tm1.b is not None else 0.0),
            covariance_matrix=gls_params_tm1.R,
        )
        x_t = x_t_dist.mean if deterministic else x_t_dist.rsample()
        z_t_dist = torch.distributions.MultivariateNormal(
            loc=matvec(gls_params_t.C, x_t)
            + (gls_params_t.d if gls_params_t.d is not None else 0.0),
            covariance_matrix=gls_params_t.Q,
        )
        z_t = z_t_dist.mean if deterministic else z_t_dist.rsample()
        return x_t, z_t, gls_params_t, rnn_state_t

    def compute_gls_params(self, rnn_inputs, u_state, u_obs):
        # There is no transition at the last time-step. We have one param less for dynamics.
        # But changed now that the respective function removes the last param.
        # This is more explicit and clear.
        shp = rnn_inputs.shape
        shp_flat = (shp[0],) + (np.prod(shp[1:-1]),) + (shp[-1],)
        rnn_output, last_rnn_state = self.rnn_switch_model(
            input=rnn_inputs.reshape(shp_flat)
        )
        rnn_output = rnn_output.reshape(shp[:-1] + (rnn_output.shape[-1],))
        gls_params = self.gls_base_parameters(
            switch=rnn_output, u_state=u_state, u_obs=u_obs,
        )
        # TODO: must return also state for forecasting, where we first filter then sample,
        #  using rnn_state[t-1]. But then this function has a bad name.
        #  Redesign this.
        return gls_params, last_rnn_state


from typing import Optional, Dict, Iterator
from gluonts.dataset.common import Dataset
from gluonts.dataset.field_names import FieldName
from gluonts.model.forecast import Forecast, SampleForecast
from gluonts.model.predictor import RepresentablePredictor
from gluonts.dataset.loader import InferenceDataLoader
from data.gluonts_nips_datasets.gluonts_nips_datasets import (
    transform_gluonts_to_pytorch,
)


class KVAEPredictor(RepresentablePredictor):
    """ wrapper to to allow make_evaluation_predictions to evaluate this model. """

    def __init__(
        self,
        model: KalmanVariationalAutoEncoder,
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
    ):
        super().__init__(
            prediction_length=prediction_length, freq=freq, lead_time=lead_time
        )
        self.model = model  # Note: does not support multi-GPU DataParallel model. as we did not implement forward.
        self.input_transform = input_transform
        self.batch_size = batch_size
        self.cardinalities = cardinalities
        self.dims = dims
        self.bias_y = bias_y
        self.factor_y = factor_y
        self.time_feat = time_feat
        # gluonts calls predict without alllowed kwargs from outside in backtest.py...
        # therefore we set this option via mutable attributes. sweet.

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
            batch.pop("seasonal_indicators")  # not used in this model.
            # 1) filter
            n_steps_filter = len(batch["y"])
            batch_filter = {
                name: data[:n_steps_filter] for name, data in batch.items()
            }
            batch_forecast = {
                name: data[n_steps_filter:]
                for name, data in batch.items()
                if name is not "y"
            }

            (
                x_filter_dist,
                z_filter_dist,
                gls_params,
                last_rnn_state,
                inv_measurement_dist,
                z,
            ) = self.model.filter_forward(
                **batch_filter
            )  # last time-step same as with smoothing.
            # 2) forecast --> sample
            z_filter = z_filter_dist.sample()
            (
                x_forecast,
                z_forecast,
                gls_params_forecast,
                rnn_state_forecast,
            ) = self.model.sample(
                x=x_filter_dist.sample()[-1],
                z=z_filter[-1],
                gls_params=Box(
                    {
                        key: val[-1] if val is not None else None
                        for key, val in gls_params.items()
                    }
                ),
                rnn_state=last_rnn_state,
                n_timesteps=self.prediction_length,
                n_batch=x_filter_dist.loc.shape[2],
                **batch_forecast,
            )
            y_forecast_dist = self.model.measurement_model(z_forecast)
            y_forecast = y_forecast_dist.sample()

            # transform forecast into GTS format\
            forecast_gts = y_forecast
            forecast_gts = (forecast_gts * self.factor_y) + self.bias_y
            forecast_gts = forecast_gts.transpose(0, 2)  # TPBF -> BPTF
            forecast_gts = forecast_gts.detach().cpu().numpy()
            forecast_gts = forecast_gts.squeeze(
                axis=-1
            )  # this is bad, but backtest requires it.

            # return iterator over both forecast object and filter objects.
            for idx_sample_in_batch, _fcst_gts, in enumerate(forecast_gts,):
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
            assert idx_sample_in_batch + 1 == len(batch_gts["forecast_start"])
