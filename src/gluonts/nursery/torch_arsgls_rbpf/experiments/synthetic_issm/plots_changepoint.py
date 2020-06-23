import os
import matplotlib.pyplot as plt
from box import Box
import numpy as np
import torch
from torch.distributions import Normal
from data.synthetic_issm import generate_synthetic_issm_changepoint
from experiments.synthetic_issm.plots import _savefig, \
    plot_state_prior_final, plot_switch_trajectories, \
    plot_filtered_states, plot_predictive_distribution


def make_state_prior_plots(model, log_paths, show=False):
    m0_groundtruth = generate_synthetic_issm_changepoint.m0_seasonality
    sigma0_groundtruth = generate_synthetic_issm_changepoint.sigma0_seasonality
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


def plot_filtered_forecasted_covariances(
        gls_params, data, norm_weights, n_steps_forecast,
        lat_noise_scale_groundtruths, obs_noise_scale_groundtruth,
        n_steps_filter_plot=0, n_groups=5,
):
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
        squeeze=False, sharex=True, sharey=False,
    )
    figQ, axsQ = plt.subplots(
        n_groups, 2, figsize=[2 * 4, n_groups * 2],
        squeeze=False, sharex=True, sharey=False,
    )
    for idx_group in range(n_groups):
        axR_samples = axsR[idx_group, 0]
        axR_aggregate = axsR[idx_group, 1]
        axQ_samples = axsQ[idx_group, 0]
        axQ_aggregate = axsQ[idx_group, 1]

        LR_gt = [lat_noise_scale_groundtruths[_t, idx_group, _t % dim_state] for
                 _t in t]
        LQ_gt = [obs_noise_scale_groundtruth[idx_group, _t % dim_state] for _t
                 in t]

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

        # for idx_data in range(len(data_indices)):
        #     axR_samples.plot(
        #         t, LR[:, :, idx_data],
        #         color=colors(idx_data), linewidth=0.25, alpha=0.5)
        #     axQ_samples.plot(
        #         t, LQ[:, :, idx_data],
        #         color=colors(idx_data), linewidth=0.25, alpha=0.5)

        LR_heatmap_grid = torch.tensor(np.linspace(0.0, 3.0, 300))
        scale = LR_heatmap_grid.max() / len(LR_heatmap_grid) * 5
        LR_normal = Normal(loc=torch.tensor(LR), scale=scale)
        LR_log_probs = \
            LR_normal.log_prob(LR_heatmap_grid[:, None, None, None])[
                ..., None].transpose(0, -1)[0]
        LR_weighted_probs = (torch.exp(LR_log_probs) * weights[..., None]).sum(
            dim=[1, 2])
        axR_samples.imshow(LR_weighted_probs.T.numpy()[::-1],
                           cmap='gray', interpolation='none', aspect="auto",
                           extent=[t[0], t[-1], LR_heatmap_grid[0],
                                   LR_heatmap_grid[-1]])

        LQ_heatmap_grid = torch.tensor(np.linspace(0.0, 0.15, 300))
        scale = LQ_heatmap_grid.max() / len(LQ_heatmap_grid) * 5
        LQ_normal = Normal(loc=torch.tensor(LQ), scale=scale)
        LQ_log_probs = \
            LQ_normal.log_prob(LQ_heatmap_grid[:, None, None, None])[
                ..., None].transpose(0, -1)[0]
        LQ_weighted_probs = (torch.exp(LQ_log_probs) * weights[..., None]).sum(
            dim=[1, 2])
        axQ_samples.imshow(LQ_weighted_probs.T.numpy()[::-1],
                           cmap='gray', interpolation='none', aspect="auto",
                           extent=[t[0], t[-1], LQ_heatmap_grid[0],
                                   LQ_heatmap_grid[-1]])

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
        # axR_samples.grid(True)
        miniR = np.min([lat_noise_scale_groundtruths[idx_gt][idx_group]
                        for idx_gt in range(len(lat_noise_scale_groundtruths))])
        maxiR = np.max([lat_noise_scale_groundtruths[idx_gt][idx_group]
                        for idx_gt in range(len(lat_noise_scale_groundtruths))])
        # axR_samples.set_ylim([miniR - 0.05, maxiR + 0.05])
        axR_aggregate.set_ylim([miniR - 0.05, maxiR + 0.05])

        axQ_aggregate.set_xlabel("t [day]")
        axQ_aggregate.set_ylabel("$Q^{1/2}$")
        axQ_aggregate.grid(True)
        axQ_samples.set_xlabel("t [day]")
        axQ_samples.set_ylabel("$Q^{1/2}$")
        # axQ_samples.grid(True)
        miniQ, maxiQ = obs_noise_scale_groundtruth.min(), obs_noise_scale_groundtruth.max()
        # axQ_samples.set_ylim([miniQ - 0.05, maxiQ + 0.05])
        axQ_aggregate.set_ylim([miniQ - 0.05, maxiQ + 0.05])

    figR.suptitle("latent noise scale", y=1.0)
    figQ.suptitle("observation noise scale", y=1.0)
    return figR, figQ, axsR, axsQ


def make_forecast_plots(epoch, model, log_paths, dims, n_steps_forecast, data,
                        show=False):
    log_norm_weights, s, m, V, mpy, Vpy, gls_params = model.filter_forecast(
        n_steps_forecast=n_steps_forecast, **data)
    # switches, states, observations, gls_params = model.sample(
    # u_switch=data['u_switch'], n_timesteps=dims.timesteps, n_batch=dims.batch)
    # gls_params = Box({name: torch.stack(val).detach().cpu().numpy() if val is not None and val[
    #     0] is not None else None
    #                   for name, val in gls_params.items()})

    norm_weights = torch.exp(log_norm_weights).detach().cpu().numpy()
    gls_params = Box(
        {name: val.detach().cpu().numpy() if val is not None else None
         for name, val in gls_params.items()})
    s = s.detach().cpu().numpy()
    m = m.detach().cpu().numpy()
    V = V.detach().cpu().numpy()
    mpy = mpy.detach().cpu().numpy()
    Vpy = Vpy.detach().cpu().numpy()
    y = data['y'].cpu()

    idx_batch, idx_particle = 0, 0
    fig, axs = plot_switch_trajectories(s=s)
    _savefig(fig=fig, fname=os.path.join(log_paths.plot,
                                         f"switch_trajectory_ep{epoch}.pdf"))

    fig, axs = plot_filtered_states(
        m=m, V=V, norm_weights=norm_weights,
        idx_batch=idx_batch, idx_particle=idx_particle)
    _savefig(fig=fig, fname=os.path.join(
        log_paths.plot, f"filtered_state_p{idx_particle}_b{idx_batch}.pdf"))
    fig, axs = plot_predictive_distribution(
        y=y, mpy=mpy, Vpy=Vpy, norm_weights=norm_weights,
        idx_batch=idx_batch, idx_particle=idx_particle)
    _savefig(fig=fig, fname=os.path.join(
        log_paths.plot, f"predictive_dist_p{idx_particle}_b{idx_batch}.pdf"))
    fig, axs = plot_filtered_states(
        m=m, V=V, norm_weights=norm_weights, idx_batch=idx_batch,
        idx_particle=None)
    _savefig(fig=fig, fname=os.path.join(
        log_paths.plot, f"filtered_state_importance_weighted_b{idx_batch}.pdf"))
    fig, axs = plot_predictive_distribution(
        y=y, mpy=mpy, Vpy=Vpy, norm_weights=norm_weights, idx_batch=idx_batch,
        idx_particle=None)
    _savefig(fig=fig, fname=os.path.join(
        log_paths.plot,
        f"predictive_dist_importance_weighted_b{idx_batch}.pdf"))

    assert generate_synthetic_issm_changepoint.sigmas.ndim == 1
    assert all(generate_synthetic_issm_changepoint.gammas_list[idx].ndim == 2
               for idx in
               range(len(generate_synthetic_issm_changepoint.gammas_list)))

    n_groups = generate_synthetic_issm_changepoint.gammas_list[0].shape[0]

    obs_noise_scale_groundtruth = np.repeat(
        generate_synthetic_issm_changepoint.sigmas[None, :], repeats=n_groups,
        axis=0)

    change_timesteps = generate_synthetic_issm_changepoint.change_timesteps
    T = dims.timesteps + n_steps_forecast
    pattern_durations = np.array(tuple(change_timesteps) + (T,)) \
                        - np.array((0,) + tuple(change_timesteps))
    gammas_list = generate_synthetic_issm_changepoint.gammas_list

    # TODO: these groundtruths are not correct (50%) for random_changepoint.
    #  Not implemented for random_changepoint.
    lat_noise_scale_groundtruths = np.concatenate([
        gammas_list[idx][None, :].repeat(duration, axis=0)
        for idx, duration in enumerate(pattern_durations)
    ], axis=0)

    figR, figQ, axsR, axsQ = plot_filtered_forecasted_covariances(
        gls_params=gls_params, norm_weights=norm_weights,
        data=data, n_steps_forecast=n_steps_forecast,
        lat_noise_scale_groundtruths=lat_noise_scale_groundtruths,
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
        lat_noise_scale_groundtruths=lat_noise_scale_groundtruths,
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
