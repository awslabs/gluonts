import os
from copy import deepcopy
import matplotlib.pyplot as plt
from box import Box
import numpy as np
import torch
from data.synthetic_issm import generate_synthetic_issm


def _load_past_params(epoch, model, log_paths):
    model = deepcopy(model)  # do not change (by reference) the original model.
    model_params = {name: [None] * epoch for name, val in
                    model.named_parameters()}
    for idx_epoch in range(epoch):
        model.load_state_dict(
            torch.load(os.path.join(log_paths.model, f"{idx_epoch}.pt")))
        model.eval()
        for name, val in model.named_parameters():
            model_params[name][idx_epoch] = val.detach().cpu().numpy()
    model_params = {name: np.stack(val, axis=0) for name, val in
                    model_params.items()}
    return model_params


def _savefig(fig, fname, bbox_inches="tight", pad_inches=0.025, dpi=150):
    # short for some default params.
    fig.savefig(fname=fname, bbox_inches=bbox_inches, pad_inches=pad_inches,
                dpi=dpi)


def plot_observations(data, n_data_plot):
    assert data["y"].shape[-1] == 1
    y = data["y"][..., 0]
    fig = plt.figure()
    plt.plot(y[:, :n_data_plot])
    plt.xlabel("t [days]")
    plt.ylabel("y")
    plt.title("observations")
    return fig


def plot_state_prior_over_training(model_params):
    m0 = model_params["state_prior_model.m"]
    sigma0 = np.exp(-model_params["state_prior_model.LVinv_logdiag"])
    fig, (ax_m, ax_sig) = plt.subplots(2, 1, squeeze=True)
    it = np.arange(m0.shape[0])
    ax_m.plot(it, m0)
    ax_m.set_ylabel("loc")
    ax_m.set_xlabel("iteration")
    ax_m.legend([f"dim {idx_dim}" for idx_dim in range(m0.shape[-1])])
    ax_sig.plot(it, sigma0)
    ax_sig.set_ylabel("scale")
    ax_sig.set_xlabel("iteration")
    ax_sig.legend([f"dim {idx_dim}" for idx_dim in range(m0.shape[-1])])
    fig.suptitle("state initial prior parameters during training", y=1.0)
    return fig, (ax_m, ax_sig)


def plot_state_noise_over_training(state_noise_scale):
    n_iter, n_switch, n_season = state_noise_scale.shape
    fig, axs = plt.subplots(
        n_switch, n_season, figsize=[n_season * 3, n_switch * 2],
        squeeze=False, sharex=True, sharey=True,
    )
    it = np.arange(n_iter)
    for idx_season in range(n_season):
        for idx_switch in range(n_switch):
            ax = axs[idx_switch, idx_season]
            ax.set_yscale("log")
            # ax.yaxis.set_major_formatter(ScalarFormatter())
            # ax.yaxis.set_major_locator(LogLocator())
            ax.plot(it, state_noise_scale[:, idx_switch, idx_season])
            ax.set_xlabel("iteration")
            ax.set_ylabel("$R^{1/2}$")
            ax.grid(True)
    fig.suptitle(
        "latent noise scale for all switches (rows) and seasons (cols)", y=1.0)
    return fig, axs


def plot_observation_noise_over_training(obs_noise_scale):
    n_iter, n_season, n_switch, n_obs = obs_noise_scale.shape
    fig, axs = plt.subplots(
        n_switch, n_season, figsize=[n_season * 3, n_switch * 2],
        squeeze=False, sharex=True, sharey=True,
    )
    it = np.arange(n_iter)
    for idx_season in range(n_season):
        for idx_switch in range(n_switch):
            ax = axs[idx_switch, idx_season]
            ax.set_yscale("log")
            # ax.yaxis.set_major_formatter(ScalarFormatter())
            # ax.yaxis.set_major_locator(LogLocator())
            ax.plot(it, obs_noise_scale[:, idx_season, idx_switch])
            ax.set_xlabel("iteration")
            ax.set_ylabel("$Q^{1/2}$")
            ax.grid(True)
    fig.suptitle(
        "observation noise scale for all switches (rows) and seasons (cols)",
        y=1.0)
    return fig, axs


def plot_switch_trajectories(s, n_data_plot=5, n_particles_plot=1,
                             idx_forecast_start=None):
    # Assume s is given as TBPS
    n_timesteps, n_particle, n_batch, n_switch = s.shape
    n_data_plot = n_batch if n_data_plot > n_batch else n_data_plot
    n_particles_plot = n_particle if n_particles_plot > n_particle else n_particles_plot
    t = np.arange(n_timesteps)

    colors = plt.cm.get_cmap("viridis", n_switch)
    fig, axs = plt.subplots(n_data_plot, figsize=[4, n_data_plot * 2],
                            squeeze=True)
    for idx_data in range(n_data_plot):
        ax = axs[idx_data]
        for idx_switch in range(n_switch):
            ax.plot(t, s[:, :n_particles_plot, idx_data, idx_switch],
                    color=colors(idx_switch), linewidth=1)
        ax.set_xlabel("t")
        ax.set_ylabel("s")
        # ax.legend([f"dim {idx_switch}" for idx_switch in range(n_switch)])
        if idx_forecast_start is not None:
            ax.axvline(idx_forecast_start, color="black")
    fig.suptitle("switch variable samples --- color indicates switch dimension",
                 y=1.0)
    return fig, axs


def plot_state_prior_final(model_params, m0_groundtruth, sigma0_groundtruth):
    if not isinstance(model_params, (tuple, list)):
        model_params = [model_params]
    n_models = len(model_params)
    m0s = [None] * n_models
    sigma0s = [None] * n_models
    for m, mps in enumerate(model_params):
        m0s[m] = mps["state_prior_model.m"]
        sigma0s[m] = np.exp(-mps["state_prior_model.LVinv_logdiag"])
    n_dims = m0s[0].shape[-1]
    x_dims = tuple(range(n_dims))
    assert all(tuple(p.shape) == (n_dims,) for p in m0s)
    assert all(tuple(p.shape) == (n_dims,) for p in sigma0s)

    fig, (ax_m, ax_sig) = plt.subplots(2, 1, squeeze=True)
    ax_m.set_ylabel("loc")
    ax_m.set_xlabel("dimension")
    ax_sig.set_ylabel("scale")
    ax_sig.set_xlabel("dimension")
    for m0, sigma0 in zip(m0s, sigma0s):
        ax_m.plot(x_dims, m0)
        ax_sig.plot(x_dims, sigma0)
        ax_m.scatter(x_dims, m0_groundtruth, marker="x")
        ax_sig.scatter(x_dims, sigma0_groundtruth, marker="x")
    ax_m.legend(["model", "ground truth"])
    ax_sig.legend(["model", "ground truth"])
    fig.suptitle("initial state prior", y=1.0)
    return fig, (ax_m, ax_sig)


def plot_filtered_forecasted_covariances(gls_params, data, norm_weights,
                                         n_steps_forecast,
                                         lat_noise_scale_groundtruth,
                                         obs_noise_scale_groundtruth,
                                         n_steps_filter_plot=0, n_groups=5):
    n_timesteps, n_particle, n_batch, dim_state, dim_state = gls_params.R.shape
    n_timesteps, n_particle, n_batch, dim_obs, dim_obs = gls_params.Q.shape
    assert dim_obs == 1
    assert dim_state == 7
    t = np.array(tuple(
        range(n_timesteps - n_steps_forecast - n_steps_filter_plot,
              n_timesteps)))
    assert t[0] >= 0

    # R
    # 1) sort batch by groups.
    # NOTE: assumes first 5 features are group!
    idx_timestep = 0  # at all time-steps same group in this experiment
    group_indices = [
        [idx_data for idx_data in range(n_batch) if data["u_static_cat"][
            idx_timestep, idx_data, idx_group] == 1] for idx_group in
        range(n_groups)]

    figR, axsR = plt.subplots(
        n_groups, 2, figsize=[2 * 4, n_groups * 2],
        squeeze=False, sharex=True, sharey=True,
    )
    figQ, axsQ = plt.subplots(
        n_groups, 2, figsize=[2 * 4, n_groups * 2],
        squeeze=False, sharex=True, sharey=True,
    )
    for idx_group in range(n_groups):
        axR_samples = axsR[idx_group, 0]
        axR_aggregate = axsR[idx_group, 1]
        axQ_samples = axsQ[idx_group, 0]
        axQ_aggregate = axsQ[idx_group, 1]

        data_indices = group_indices[idx_group]
        # NOTE: Do not re-write this with np.stack, indexing a specific t,
        # numpy has a weird slicing bug if there are 2 dimensions before the slicing.
        # just try: np.arange(27).reshape([1,3,3,3])[0, :, [0, 1], 0].shape
        # Note that this works fine in torch.
        R = np.concatenate([gls_params.R[t:t + 1, :, data_indices,
                            t % dim_state, t % dim_state]
                            for t in t], axis=0)
        Q = np.concatenate([gls_params.Q[t:t + 1, :, data_indices, 0, 0]
                            for t in t], axis=0)
        LR = R ** 0.5  # lat noise scale
        LQ = Q ** 0.5  # obs noise scale
        weights = norm_weights[t, ...][
            ..., data_indices]  # cannot index simultanously

        LR_mean = np.mean(np.sum(weights * LR, axis=1), axis=1)
        LQ_mean = np.mean(np.sum(weights * LQ, axis=1), axis=1)
        LR_std = np.std(np.sum(weights * LR, axis=1), axis=1)
        LQ_std = np.std(np.sum(weights * LQ, axis=1), axis=1)

        LR_gt = [lat_noise_scale_groundtruth[idx_group, t % dim_state] for t in
                 t]
        LQ_gt = [obs_noise_scale_groundtruth[idx_group, t % dim_state] for t in
                 t]

        colors = plt.cm.get_cmap("viridis", len(data_indices))
        if n_steps_filter_plot != 0:
            axR_aggregate.axvline(t[-1] - n_steps_forecast, color="black",
                                  linestyle="--")
            axR_samples.axvline(t[-1] - n_steps_forecast, color="black",
                                linestyle="--")
            axQ_aggregate.axvline(t[-1] - n_steps_forecast, color="black",
                                  linestyle="--")
            axQ_samples.axvline(t[-1] - n_steps_forecast, color="black",
                                linestyle="--")

        for idx_data in range(len(data_indices)):
            axR_samples.plot(
                t, LR[:, :, idx_data],
                color=colors(idx_data), linewidth=0.25, alpha=0.5)
            axQ_samples.plot(
                t, LQ[:, :, idx_data],
                color=colors(idx_data), linewidth=0.25, alpha=0.5)

        axR_aggregate.plot(t, LR_mean)
        axR_aggregate.fill_between(
            t, LR_mean - 3 * LR_std, LR_mean + 3 * LR_std,
            alpha=0.25,
        )
        axR_aggregate.scatter(t, LR_gt, marker="x", s=5)
        axQ_aggregate.plot(t, LQ_mean)
        axQ_aggregate.fill_between(
            t, LQ_mean - 3 * LQ_std, LQ_mean + 3 * LQ_std,
            alpha=0.25,
        )
        axQ_aggregate.scatter(t, LQ_gt, marker="x", s=5)

        axR_aggregate.set_xlabel("t [day]")
        axR_aggregate.set_ylabel("$R^{1/2}$")
        axR_aggregate.grid(True)
        axR_samples.set_xlabel("t [day]")
        axR_samples.set_ylabel("$R^{1/2}$")
        axR_samples.grid(True)
        miniR, maxiR = lat_noise_scale_groundtruth.min(), lat_noise_scale_groundtruth.max()
        axR_samples.set_ylim([miniR - 0.05, maxiR + 0.05])
        axR_aggregate.set_ylim([miniR - 0.05, maxiR + 0.05])

        axQ_aggregate.set_xlabel("t [day]")
        axQ_aggregate.set_ylabel("$Q^{1/2}$")
        axQ_aggregate.grid(True)
        axQ_samples.set_xlabel("t [day]")
        axQ_samples.set_ylabel("$Q^{1/2}$")
        axQ_samples.grid(True)
        miniQ, maxiQ = obs_noise_scale_groundtruth.min(), obs_noise_scale_groundtruth.max()
        axQ_samples.set_ylim([miniQ - 0.05, maxiQ + 0.05])
        axQ_aggregate.set_ylim([miniQ - 0.05, maxiQ + 0.05])

    figR.suptitle("latent noise scale: left samples, right mean and 3*std",
                  y=1.0)
    figQ.suptitle("observation noise scale: left samples, right mean and 3*std",
                  y=1.0)
    return figR, figQ, axsR, axsQ


# ***** functions that make multiple plots *****
def make_params_over_training_plots(epoch, model, log_paths, show=False):
    model_params = _load_past_params(epoch=epoch, model=model,
                                     log_paths=log_paths)
    state_noise_scale = np.exp(
        -model_params["gls_base_parameters.LRinv_logdiag"])
    obs_noise_scale = np.exp(-model_params["gls_base_parameters.LQinv_logdiag"])
    if obs_noise_scale.ndim == 3:  # add switch dimension
        obs_noise_scale = obs_noise_scale[:, None, :, :]

    fig, axs = plot_state_prior_over_training(model_params=model_params)
    _savefig(fig=fig, fname=os.path.join(log_paths.plot,
                                         "state_prior_during_training.pdf"))

    fig, axs = plot_state_noise_over_training(
        state_noise_scale=state_noise_scale)
    _savefig(fig=fig, fname=os.path.join(log_paths.plot,
                                         "state_noise_during_training.pdf"))

    fig, axs = plot_observation_noise_over_training(
        obs_noise_scale=obs_noise_scale)
    _savefig(fig=fig, fname=os.path.join(log_paths.plot,
                                         "observation_noise_during_training.pdf"))

    if show:
        plt.show()
    plt.close()


def make_state_prior_plots(model, log_paths, show=False):
    m0_groundtruth = generate_synthetic_issm.m0_seasonality
    sigma0_groundtruth = generate_synthetic_issm.sigma0_seasonality
    model_params = {name: param.detach().cpu().numpy() for name, param in
                    model.named_parameters()}
    fig, axs = plot_state_prior_final(model_params=model_params,
                                      m0_groundtruth=m0_groundtruth,
                                      sigma0_groundtruth=sigma0_groundtruth)
    _savefig(fig=fig,
             fname=os.path.join(log_paths.plot, "state_prior_final.pdf"))
    if show:
        plt.show()
    plt.close()


def make_forecast_plots(epoch, model, log_paths, dims, n_steps_forecast, data,
                        show=False):
    log_norm_weights, s, m, V, mpy, Vpy, gls_params = model.filter_forecast(
        n_particle=dims.particle, n_steps_forecast=n_steps_forecast,
        **data)
    norm_weights = torch.exp(log_norm_weights).detach().cpu().numpy()
    gls_params = Box(
        {name: val.detach().cpu().numpy() if val is not None else None
         for name, val in gls_params.items()})
    s = s.detach().cpu().numpy()

    fig, axs = plot_switch_trajectories(s=s)
    _savefig(fig=fig, fname=os.path.join(log_paths.plot,
                                         f"switch_trajectory_ep{epoch}.pdf"))

    assert generate_synthetic_issm.sigmas.ndim == 1, generate_synthetic_issm.gammas.ndim == 2
    n_groups = generate_synthetic_issm.gammas.shape[0]
    lat_noise_scale_groundtruth = generate_synthetic_issm.gammas
    obs_noise_scale_groundtruth = np.repeat(
        generate_synthetic_issm.sigmas[None, :], repeats=n_groups, axis=0)

    figR, figQ, axsR, axsQ = plot_filtered_forecasted_covariances(
        gls_params=gls_params, norm_weights=norm_weights,
        data=data, n_steps_forecast=n_steps_forecast,
        lat_noise_scale_groundtruth=lat_noise_scale_groundtruth,
        obs_noise_scale_groundtruth=obs_noise_scale_groundtruth,
        n_steps_filter_plot=dims.timesteps,
    )
    _savefig(fig=figR, fname=os.path.join(log_paths.plot,
                                          f"filtered_forecasted_R_ep{epoch}.pdf"))
    _savefig(fig=figQ, fname=os.path.join(log_paths.plot,
                                          f"filtered_forecasted_Q_ep{epoch}.pdf"))

    figR, figQ, axsR, axsQ = plot_filtered_forecasted_covariances(
        gls_params=gls_params, norm_weights=norm_weights,
        data=data, n_steps_forecast=n_steps_forecast,
        lat_noise_scale_groundtruth=lat_noise_scale_groundtruth,
        obs_noise_scale_groundtruth=obs_noise_scale_groundtruth,
        n_steps_filter_plot=0,
    )
    _savefig(fig=figR,
             fname=os.path.join(log_paths.plot, f"forecasted_R_ep{epoch}.pdf"))
    _savefig(fig=figQ,
             fname=os.path.join(log_paths.plot, f"forecasted_Q_ep{epoch}.pdf"))

    if show:
        plt.show()
    plt.close()


def plot_predictive_distribution(y, mpy, Vpy, norm_weights, idx_batch,
                                 idx_particle=None):
    t = np.arange(mpy.shape[0])
    if idx_particle is None:
        _mpy = (norm_weights[:, :, idx_batch] * mpy[:, :, idx_batch, 0]).sum(
            axis=1)
        _Vpy = np.sum(
            (norm_weights[:, :, idx_batch] * Vpy[:, :, idx_batch, 0, 0]),
            axis=1) \
               + np.sum(
            (norm_weights[:, :, idx_batch] * mpy[:, :, idx_batch, 0] ** 2),
            axis=1) \
               - _mpy ** 2
        fig, axs = plt.subplots(1, 1)
        ax_p, ax_w = axs, None
    else:
        _mpy = mpy[:, idx_particle, idx_batch, 0]
        _Vpy = Vpy[:, idx_particle, idx_batch, 0, 0]
        fig, axs = plt.subplots(2, 1)
        ax_p, ax_w = axs

    ax_p.plot(t, _mpy)
    ax_p.fill_between(t, _mpy - 3 * _Vpy ** 0.5, _mpy + 3 * _Vpy ** 0.5,
                      alpha=0.25)
    # axs[0].legend("norm importance weight")
    ax_p.scatter(t[:y.shape[0]], y[:, idx_batch, 0], marker="x")
    ax_p.legend(["mean", "3 std", "true obs"])
    ax_p.set_xlabel("t")
    ax_p.set_ylabel("y")
    # plt.ylim([-5, 15])
    if idx_particle is None:
        ax_p.set_title(f"predictive distributions - "
                       f"importance-weighted, data idx {idx_batch}")
    else:
        ax_p.set_title(f"predictive distributions - "
                       f"particle {idx_particle}, data idx {idx_batch}")
    if idx_particle is not None:
        ax_w.plot(t, norm_weights[:, idx_particle, idx_batch], color='black')
        ax_w.set_xlabel("t")
        ax_w.set_ylabel("weight")
    return fig, axs


def plot_filtered_states(m, V, norm_weights, idx_batch, idx_particle=None):
    n_state = m.shape[-1]
    colors = plt.cm.get_cmap("viridis", n_state)
    t = np.arange(m.shape[0])
    if idx_particle is None:
        fig, axs = plt.subplots(1, 1)
        ax_f, ax_w = axs, None
    else:
        fig, axs = plt.subplots(2, 1)
        ax_f, ax_w = axs
    for idx_state in range(n_state):
        if idx_particle is None:
            _m = (norm_weights[:, :, idx_batch] * m[:, :, idx_batch,
                                                  idx_state]).sum(axis=1)
            _V = np.sum((norm_weights[:, :, idx_batch] * V[:, :, idx_batch,
                                                         idx_state, idx_state]),
                        axis=1) \
                 + np.sum((norm_weights[:, :, idx_batch] * m[:, :, idx_batch,
                                                           idx_state] ** 2),
                          axis=1) \
                 - _m ** 2
        else:
            _m = m[:, idx_particle, idx_batch, idx_state]
            _V = V[:, idx_particle, idx_batch, idx_state, idx_state]

        ax_f.plot(_m, color=colors(idx_state))
        ax_f.fill_between(t, _m - 3 * _V ** 0.5, _m + 3 * _V ** 0.5,
                          alpha=0.25, color=colors(idx_state))
        if idx_particle is not None:
            ax_w.plot(t, norm_weights[:, idx_particle, idx_batch],
                      color="black")
    ax_f.legend([f"dim {idx_state}" for idx_state in range(n_state)])
    if idx_particle is None:
        ax_f.set_title(f"filtered and forecasted states - "
                       f"importance-weighted, data idx {idx_batch}")
    else:
        ax_f.set_title(f"filtered and forecasted states - "
                       f"particle {idx_particle}, data idx {idx_batch}")
    ax_f.set_xlabel("t")
    ax_f.set_ylabel("x")
    if idx_particle is not None:
        ax_w.set_xlabel("t")
        ax_w.set_ylabel("weight")
    return fig, axs


if __name__ == "__main__":
    from src.data.synthetic_issm.synthetic_issm_loader import \
        create_trainset_loader, \
        gluonts_batch_to_train_pytorch
    from src.experiments.synthetic_issm.config import config

    train_loader = create_trainset_loader(
        n_data_per_group=config.n_data_per_group, batch_size=config.dims.batch)
    data = next(iter(train_loader))
    data = gluonts_batch_to_train_pytorch(batch=data, device="cpu",
                                          dtype=torch.float32,
                                          dims=config.dims)
    plot_observations(data=data, n_data_plot=5)
