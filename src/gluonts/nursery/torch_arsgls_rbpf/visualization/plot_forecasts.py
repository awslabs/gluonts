import matplotlib.pyplot as plt
import numpy as np
import torch

from torch_extensions.ops import matmul, batch_diag, batch_diag_matrix


def make_val_plots(model, data, idx_timeseries, n_steps_forecast, savepath,
                   y_gt=None,
                   idx_particle=None, show=False):
    module = model.module if hasattr(model, "module") else model
    device = module.state_prior_model.m.device
    # data = next(iter(test_loader))
    data = {name: val.to(device) for name, val in data.items()}
    data = {key: val[:, idx_timeseries: idx_timeseries + 1] for key, val in
            data.items()}
    y = data['y']
    data['y'] = y[:-n_steps_forecast, :]

    if y_gt is not None:
        y_plot = torch.cat(
            [data["y"],
             y_gt[-n_steps_forecast:, idx_timeseries: idx_timeseries + 1:]],
            dim=0)
    else:
        y_plot = y

    # losses_time_batch_wise, y_forecast, mpy_filter, Vpy_filter = model(
    #     **data, make_forecast=True, n_steps_forecast=config.n_steps_forecast)
    log_norm_weights_trajectory, s_trajectory, m_trajectory, V_trajectory, \
    mpy_trajectory, Vpy_trajectory, gls_params_trajectory = module.filter_forecast(
        n_steps_forecast=n_steps_forecast, **data)
    norm_weights_trajectory = torch.exp(log_norm_weights_trajectory)

    fig, axs = plot_predictive_distribution(
        y=y_plot.detach(),
        mpy=mpy_trajectory.detach(),
        Vpy=Vpy_trajectory.detach(),
        norm_weights=norm_weights_trajectory.detach(),
        n_steps_forecast=n_steps_forecast,
        idx_timeseries=0,
        # we already provide only data for series with idx_timeseries.
        idx_particle=idx_particle,
        show=show,
        savepath=savepath,
    )
    plt.close(fig)


# TODO: everything duplicated except filter_forecast does not return m, V
def make_val_plots_auxiliary(model, data, idx_timeseries, n_steps_forecast,
                             savepath,
                             y_gt=None, idx_particle=None, show=False):
    module = model.module if hasattr(model, "module") else model
    device = module.state_prior_model.m.device
    # data = next(iter(test_loader))
    data = {name: val.to(device) for name, val in data.items()}
    data = {key: val[:, idx_timeseries: idx_timeseries + 1] for key, val in
            data.items()}
    y = data['y']
    data['y'] = y[:-n_steps_forecast, :]

    if y_gt is not None:
        y_plot = torch.cat([
            data["y"],
            y_gt[-n_steps_forecast:, idx_timeseries: idx_timeseries + 1:]
        ], dim=0)
    else:
        y_plot = y

    # losses_time_batch_wise, y_forecast, mpy_filter, Vpy_filter = model(
    #     **data, make_forecast=True, n_steps_forecast=config.n_steps_forecast)
    log_norm_weights_trajectory, s_trajectory, z_trajectory, \
    y_trajectory_dist, gls_params_trajectory = module.filter_forecast(
        n_steps_forecast=n_steps_forecast, **data)
    mpy_trajectory = y_trajectory_dist.mean
    Vpy_trajectory = batch_diag_matrix(y_trajectory_dist.variance)
    norm_weights_trajectory = torch.exp(log_norm_weights_trajectory)

    fig, axs = plot_predictive_distribution(
        y=y_plot.detach(),
        mpy=mpy_trajectory.detach(),
        Vpy=Vpy_trajectory.detach(),
        norm_weights=norm_weights_trajectory.detach(),
        n_steps_forecast=n_steps_forecast,
        idx_timeseries=0,  # we already provide data for 1 time-series only.
        idx_particle=idx_particle,
        show=show,
        savepath=savepath,
    )
    plt.close(fig)


def _savefig(fig, fname, bbox_inches="tight", pad_inches=0.025, dpi=150):
    # short for some default params.
    fig.savefig(
        fname=fname, bbox_inches=bbox_inches, pad_inches=pad_inches, dpi=dpi)


def plot_predictive_distribution(y, mpy, Vpy, norm_weights,
                                 n_steps_forecast, idx_timeseries,
                                 idx_particle=None,
                                 show=False, savepath=None):
    t = np.arange(mpy.shape[0])
    if idx_particle is None:  # compute mean and std of mixture of Gaussians.
        w = norm_weights[:, :, idx_timeseries]
        m = mpy[:, :, idx_timeseries]
        V = Vpy[:, :, idx_timeseries]
        _mpy = (w.T * m.T).T.sum(dim=1, keepdims=True)
        dy = (m - _mpy)[..., None]
        _Vpy = (w.T * V.T).T.sum(dim=1, keepdims=True) \
               + (w.T * matmul(dy, dy.transpose(-1, -2)).T).T.sum(
            dim=1, keepdim=True)
        locs = _mpy.squeeze(dim=1).detach().cpu().numpy()
        scales = torch.sqrt(batch_diag(_Vpy)).squeeze(
            dim=1).detach().cpu().numpy()
        targets = y[:, idx_timeseries].detach().cpu().numpy()

        fig, axs = plt.subplots(1, 1, figsize=[6, 3])
        ax_p, ax_w = axs, None
    else:  # take the predictive only of a specific particle
        _mpy = mpy[:, idx_particle, idx_timeseries]
        _Vpy = Vpy[:, idx_particle, idx_timeseries]
        locs = _mpy.detach().cpu().numpy()
        scales = torch.sqrt(batch_diag(_Vpy)).detach().cpu().numpy()
        targets = y[:, idx_timeseries].detach().cpu().numpy()

        fig, axs = plt.subplots(2, 1)
        ax_p, ax_w = axs

    dims = locs.shape[-1]
    assert dims <= 5

    for dim in range(dims):
        color = next(ax_p._get_lines.prop_cycler)['color']
        loc, scale = locs[..., dim], scales[..., dim]
        target = targets[:, dim]
        ax_p.plot(t, loc, color=color)
        ax_p.fill_between(t, loc - 3 * scale, loc + 3 * scale, alpha=0.25,
                          color=color)
        ax_p.scatter(t[:len(target)], target, marker="o",
                     color=color if dims > 1 else "black",
                     s=10)
        ax_p.set_xlabel("t")
        ax_p.set_ylabel("y")
    ax_p.axvline(len(t) - n_steps_forecast, color="black", linestyle="--")
    # ax_p.legend(["mean", "3 std", "target"])
    # plt.ylim([-5, 15])
    # if idx_particle is None:
    #     ax_p.set_title(f"predictive distributions - "
    #                    f"importance-weighted, data idx {idx_timeseries}")
    # else:
    #     ax_p.set_title(f"predictive distributions - "
    #                    f"particle {idx_particle}, data idx {idx_timeseries}")
    if idx_particle is not None:
        ax_w.plot(t, norm_weights[:, idx_particle,
                     idx_timeseries].detach().cpu().numpy(),
                  color='black')
        ax_w.set_xlabel("t")
        ax_w.set_ylabel("weight")
    if show:
        fig.show()
    if savepath:
        _savefig(fig=fig, fname=savepath)
    return fig, axs


def make_val_plots_pendulum(model, data, idx_timeseries, n_steps_forecast,
                            savepath, y_gt=None,
                            idx_particle=None, show=False):
    module = model.module if hasattr(model, "module") else model
    device = module.state_prior_model.m.device
    # data = next(iter(test_loader))
    data = {name: val.to(device) for name, val in data.items()}
    data = {key: val[:, idx_timeseries: idx_timeseries + 1] for key, val in
            data.items()}
    y = data['y']
    data['y'] = y[:-n_steps_forecast, :]

    if y_gt is not None:
        y_plot = torch.cat([
            data["y"],
            y_gt[-n_steps_forecast:, idx_timeseries: idx_timeseries + 1:]
        ], dim=0)
    else:
        y_plot = y

    # losses_time_batch_wise, y_forecast, mpy_filter, Vpy_filter = model(
    #     **data, make_forecast=True, n_steps_forecast=config.n_steps_forecast)
    log_norm_weights_trajectory, s_trajectory, m_trajectory, V_trajectory, \
    mpy_trajectory, Vpy_trajectory, gls_params_trajectory = module.filter_forecast(
        n_steps_forecast=n_steps_forecast, **data)
    norm_weights_trajectory = torch.exp(log_norm_weights_trajectory)

    fig, axs = plot_predictive_distribution_pendulum(
        y=y_plot.detach(),
        mpy=mpy_trajectory.detach(),
        Vpy=Vpy_trajectory.detach(),
        norm_weights=norm_weights_trajectory.detach(),
        n_steps_forecast=n_steps_forecast,
        idx_timeseries=0,  # we already provide data for one time-series only.
        idx_particle=idx_particle,
        show=show,
        savepath=savepath,
    )
    plt.close(fig)


def plot_predictive_distribution_pendulum(y, mpy, Vpy, norm_weights,
                                          n_steps_forecast, idx_timeseries,
                                          idx_particle=None,
                                          show=False, savepath=None):
    t = np.arange(mpy.shape[0])
    if idx_particle is None:  # compute mean and std of mixture of Gaussians.
        w = norm_weights[:, :, idx_timeseries]
        m = mpy[:, :, idx_timeseries]
        V = Vpy[:, :, idx_timeseries]
        _mpy = (w.T * m.T).T.sum(dim=1, keepdims=True)
        dy = (m - _mpy)[..., None]
        _Vpy = (w.T * V.T).T.sum(dim=1, keepdims=True) \
               + (w.T * matmul(dy, dy.transpose(-1, -2)).T).T.sum(
            dim=1, keepdim=True)
        locs = _mpy.squeeze(dim=1).detach().cpu().numpy()
        scales = torch.sqrt(batch_diag(_Vpy)).squeeze(
            dim=1).detach().cpu().numpy()
        targets = y[:, idx_timeseries].detach().cpu().numpy()

        fig, axs = plt.subplots(1, 1, figsize=[6, 3])
        ax_p, ax_w = axs, None
    else:  # take the predictive only of a specific particle
        _mpy = mpy[:, idx_particle, idx_timeseries]
        _Vpy = Vpy[:, idx_particle, idx_timeseries]
        locs = _mpy.detach().cpu().numpy()
        scales = torch.sqrt(batch_diag(_Vpy)).detach().cpu().numpy()
        targets = y[:, idx_timeseries].detach().cpu().numpy()

        fig, axs = plt.subplots(2, 1)
        ax_p, ax_w = axs

    dims = locs.shape[-1]
    assert dims <= 5

    for dim in range(dims):
        color = next(ax_p._get_lines.prop_cycler)['color']
        loc, scale = locs[..., dim], scales[..., dim]
        target = targets[:, dim]
        ax_p.plot(t, loc, color=color, linewidth=1.25)
        ax_p.fill_between(t, loc - 3 * scale, loc + 3 * scale, alpha=0.25,
                          color=color)
        ax_p.scatter(t[:-n_steps_forecast], target[:-n_steps_forecast],
                     marker="o",
                     color=color if dims > 1 else "black", s=10)
        ax_p.plot(t[-n_steps_forecast:], target[-n_steps_forecast:],
                  color=color if dims > 1 else "black", linestyle="--",
                  linewidth=2.5)
        ax_p.set_xlabel("t")
        ax_p.set_ylabel("y")
        # ax_p.set_ylim([-1.0, 1.0])
    ax_p.axvline(len(t) - n_steps_forecast, color="black", linestyle="--")

    if idx_particle is not None:
        ax_w.plot(t, norm_weights[:, idx_particle,
                     idx_timeseries].detach().cpu().numpy(),
                  color='black')
        ax_w.set_xlabel("t")
        ax_w.set_ylabel("weight")
    if show:
        fig.show()
    if savepath:
        _savefig(fig=fig, fname=savepath)
    return fig, axs
