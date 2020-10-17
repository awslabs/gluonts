import matplotlib.pyplot as plt
import numpy as np
import torch

from torch_extensions.ops import matmul, batch_diag, batch_diag_matrix
from inference.smc.normalize import normalize_log_weights


def _savefig(fig, fname, bbox_inches="tight", pad_inches=0.025, dpi=150):
    """ short for some default save fig params. """
    fig.savefig(
        fname=fname, bbox_inches=bbox_inches, pad_inches=pad_inches, dpi=dpi,
    )


def make_val_plots_univariate(
    model,
    data,
    idxs_ts,
    n_steps_forecast,
    savepath,
    marginalize_states=False,
    future_target_groundtruth=None,
    idx_particle=None,
    show=False,
):
    device = model.device
    data = {name: val.to(device) for name, val in data.items()}
    future_target = data.pop("future_target")

    future_target_plot = (
        future_target_groundtruth
        if future_target_groundtruth is not None
        else future_target
    )
    y_plot = torch.cat([data["past_target"], future_target_plot])

    predictions_filtered, predictions_forecast = model(
        **data,
        n_steps_forecast=n_steps_forecast,
        marginalize_states=marginalize_states,
    )
    predictions = predictions_filtered + predictions_forecast
    if isinstance(predictions[0].emissions, torch.distributions.Distribution):
        mpy_trajectory = torch.stack([p.emissions.mean for p in predictions])
        # using distribution.variance (i.e. diagonal) for predictive variance.
        # We dont need covariances for plots. And this allows non-Gaussian.
        Vpy_trajectory = batch_diag_matrix(
            torch.stack([p.emissions.variance for p in predictions])
        )
    elif isinstance(predictions[0].emissions, torch.Tensor):
        mpy_trajectory = torch.stack([p.emissions for p in predictions])
        Vpy_trajectory = batch_diag_matrix(torch.zeros_like(mpy_trajectory))
    else:
        raise ValueError(
            f"Unexpected emission type: {type(predictions[0].emissions)}",
        )
    log_weights = torch.stack([p.latents.log_weights for p in predictions])
    norm_weights_trajectory = torch.exp(normalize_log_weights(log_weights))

    for idx_ts in idxs_ts:
        fig, axs = plot_predictive_distribution(
            y=y_plot.detach(),
            mpy=mpy_trajectory.detach(),
            Vpy=Vpy_trajectory.detach(),
            norm_weights=norm_weights_trajectory.detach(),
            n_steps_forecast=n_steps_forecast,
            idx_timeseries=idx_ts,
            idx_particle=idx_particle,
            show=show,
            savepath=f"{savepath}_b{idx_ts}.pdf",
        )
        plt.close(fig)


def plot_predictive_distribution(
    y,
    mpy,
    Vpy,
    norm_weights,
    n_steps_forecast,
    idx_timeseries,
    idx_particle=None,
    show=False,
    savepath=None,
):
    t = np.arange(mpy.shape[0])
    if idx_particle is None:  # compute mean and std of mixture of Gaussians.
        w = norm_weights[:, :, idx_timeseries]
        m = mpy[:, :, idx_timeseries]
        V = Vpy[:, :, idx_timeseries]
        _mpy = (w.T * m.T).T.sum(dim=1, keepdims=True)
        dy = (m - _mpy)[..., None]
        _Vpy = (w.T * V.T).T.sum(dim=1, keepdims=True) + (
            w.T * matmul(dy, dy.transpose(-1, -2)).T
        ).T.sum(dim=1, keepdim=True)
        locs = _mpy.squeeze(dim=1).detach().cpu().numpy()
        scales = (
            torch.sqrt(batch_diag(_Vpy)).squeeze(dim=1).detach().cpu().numpy()
        )
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
        color = next(ax_p._get_lines.prop_cycler)["color"]
        loc, scale = locs[..., dim], scales[..., dim]
        target = targets[:, dim]
        ax_p.plot(t, loc, color=color, linewidth=1.25)
        ax_p.fill_between(
            t, loc - 3 * scale, loc + 3 * scale, alpha=0.25, color=color
        )
        ax_p.scatter(
            t[:-n_steps_forecast],
            target[:-n_steps_forecast],
            marker="o",
            color=color if dims > 1 else "black",
            s=10,
        )
        ax_p.plot(
            t[-n_steps_forecast:],
            target[-n_steps_forecast:],
            color=color if dims > 1 else "black",
            linestyle="--",
            linewidth=2.5,
        )
        ax_p.set_xlabel("t")
        ax_p.set_ylabel("y")
        # ax_p.set_ylim([-1.0, 1.0])
    ax_p.axvline(len(t) - n_steps_forecast, color="black", linestyle="--")

    if idx_particle is not None:
        ax_w.plot(
            t,
            norm_weights[:, idx_particle, idx_timeseries]
            .detach()
            .cpu()
            .numpy(),
            color="black",
        )
        ax_w.set_xlabel("t")
        ax_w.set_ylabel("weight")
    if show:
        fig.show()
    if savepath:
        _savefig(fig=fig, fname=savepath)
    return fig, axs
