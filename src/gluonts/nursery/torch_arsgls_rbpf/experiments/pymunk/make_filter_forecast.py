import numpy as np
import torch
from box import Box


def make_kvae_filter_forecast(model, data_filter, prediction_length, n_particle,
                              deterministic):
    n_particle_prev = model.n_particle
    model.n_particle = n_particle if not deterministic else 1

    # 1) filter
    x_filter_dist, z_filter_dist, gls_params, last_rnn_state, inv_measurement_dist, z = \
        model.smooth_forward_backward(**data_filter)

    # take final state.
    z_filter = z_filter_dist.mean if deterministic else z_filter_dist.sample()
    x_filter = x_filter_dist.mean if deterministic else x_filter_dist.sample()

    # 2) forecast --> sample
    x_forecast, z_forecast, gls_params_forecast, rnn_state_forecast = model.sample(
        x=x_filter[-1],
        z=z_filter[-1],
        gls_params=Box({k: v[-1] if v is not None else None for k, v in
                        gls_params.items()}),
        rnn_state=last_rnn_state,
        n_timesteps=prediction_length,
        n_batch=x_filter_dist.loc.shape[2],
        deterministic=deterministic,
    )

    y_filter_dist = model.measurement_model(z_filter)
    y_forecast_dist = model.measurement_model(z_forecast)
    y_filter = y_filter_dist.mean if deterministic else y_filter_dist.sample()
    y_forecast = y_forecast_dist.mean if deterministic else y_forecast_dist.sample()

    # for plotting and consistency, fake uniform importance weights (even though KVAE has none)
    log_norm_weights_filter = torch.zeros_like(y_filter[..., 0]) - np.log(
        n_particle)
    log_norm_weights_forecast = torch.zeros_like(y_forecast[..., 0]) - np.log(
        n_particle)

    model.n_particle = n_particle_prev
    return z_filter, z_forecast, y_filter, y_forecast, log_norm_weights_filter, log_norm_weights_forecast


def make_arsgls_filter_forecast(model, data_filter, prediction_length,
                                n_particle, deterministic):
    n_particle_prev = model.n_particle
    # even if deterministic, as we start filtering with particles.
    # Only forecast predicted deterministically -> This is handled in filter_forecast.
    model.n_particle = n_particle

    # filter and forecast in one function for ARSGLS.
    log_norm_weights_trajectory, s_trajectory, z_trajectory, \
    y_trajectory_dist, gls_params_trajectory = model.filter_forecast(
        **data_filter,
        n_steps_forecast=prediction_length,
        deterministic=deterministic,
    )

    def split_filter_forecast(trajectory):
        return trajectory[:-prediction_length], trajectory[-prediction_length:]

    log_norm_weights_filter, log_norm_weights_forecast = split_filter_forecast(
        log_norm_weights_trajectory)
    z_filter, z_forecast = split_filter_forecast(z_trajectory)
    y_trajectory = y_trajectory_dist.mean if deterministic else y_trajectory_dist.sample()
    if deterministic:
        norm_weights_filter = torch.exp(log_norm_weights_filter)
        y_filter, y_forecast = split_filter_forecast(y_trajectory)

        # make weighted avg in filter range
        z_filter = (z_filter * norm_weights_filter[..., None]).sum(
            dim=1, keepdim=True)
        y_filter = (y_filter * norm_weights_filter[..., None]).sum(
            dim=1, keepdim=True)

        # pick one of n identical particles in forecast range
        z_forecast = z_forecast[:, :1]
        y_forecast = y_forecast[:, :1]
    else:
        y_filter, y_forecast = split_filter_forecast(y_trajectory)

    model.n_particle = n_particle_prev
    return z_filter, z_forecast, y_filter, y_forecast, log_norm_weights_filter, log_norm_weights_forecast
