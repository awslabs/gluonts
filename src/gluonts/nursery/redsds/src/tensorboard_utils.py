import io
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white")

color_names = ["red",
               "windows blue",
               "amber",
               "faded green",
               "dusty purple",
               "orange",
               "clay",
               "pink",
               "greyish",
               "light cyan",
               "steel blue",
               "pastel purple",
               "mint",
               "salmon",
               "plum",
               "dark orange",
               "sea blue",
               "neon purple",
               "emerald",
               "denim",
               "peacock blue"]

colors = sns.xkcd_palette(color_names)


def plot_to_image(figure):
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(figure)
    buf.seek(0)
    image = Image.open(buf)
    return np.array(image)


def show_time_series(fig_size,
                     inputs,
                     reconstructed_inputs,
                     segmentation,
                     true_segmentation=None,
                     fig_title=None,
                     ylim=None):
    def _plot_segments(segmentation, ymin, ymax):
        s_seq = np.squeeze(segmentation)
        z_cps = np.concatenate(
            ([0], np.where(np.diff(s_seq))[0]+1, [s_seq.size]))
        for start, stop in zip(z_cps[:-1], z_cps[1:]):
            stop = min(s_seq.size, stop+1)
            ax.axvspan(
                start, stop-1, ymin=ymin, ymax=ymax,
                alpha=.8, facecolor=colors[s_seq[start]])

    assert reconstructed_inputs.shape == inputs.shape
    fig = plt.figure(figsize=fig_size)
    if fig_title:
        plt.title(fig_title)
    ax = fig.gca()
    inputs = inputs.reshape(-1, inputs.shape[-1])
    reconstructed_inputs = reconstructed_inputs.reshape(
        -1, reconstructed_inputs.shape[-1])
    for i in range(inputs.shape[-1]):
        ax.scatter(
            np.arange(inputs.shape[0]), inputs[:, i],
            marker='x',
            color=colors[i],
            s=20)
        ax.plot(
            np.arange(inputs.shape[0]),
            reconstructed_inputs[:, i],
            lw=1,
            color=colors[i])
    _plot_segments(segmentation, ymin=0, ymax=0.05)
    if true_segmentation is not None:
        _plot_segments(true_segmentation, ymin=0.05, ymax=0.1)
    ax.set_xlabel('T')
    if ylim is not None:
        ax.set_ylim(ylim)
    return fig


def show_duration_dists(fig_size,
                        rho,
                        fig_title=None):
    K, d_max = rho.shape
    fig, axn = plt.subplots(nrows=K, figsize=fig_size)
    if fig_title:
        plt.title(fig_title)
    for k in range(K):
        axn[k].bar(np.arange(d_max), rho[k])
        # axn[k].set_ylim((0., 1.))
        axn[k].set_ylabel(f'State {k}')
        axn[k].set_xticks(np.arange(d_max))
    return fig


def show_time_series_sample(fig_size,
                            inputs,
                            fig_title=None,
                            ylim=None):
    fig = plt.figure(figsize=fig_size)
    if fig_title:
        plt.title(fig_title)
    ax = fig.gca()
    inputs = inputs.reshape(-1, inputs.shape[-1])
    for i in range(inputs.shape[-1]):
        ax.scatter(
            np.arange(inputs.shape[0]), inputs[:, i],
            marker='x',
            color=colors[i],
            s=20)
        ax.plot(
            np.arange(inputs.shape[0]),
            inputs[:, i],
            lw=1,
            color=colors[i])
    ax.set_xlabel('T')
    if ylim is not None:
        ax.set_ylim(ylim)
    return fig


def show_time_series_forecast(
        fig_size,
        inputs,
        rec_with_forecast,
        context_length,
        prediction_length,
        prediction_intervals=(50.0, 90.0),
        show_mean=False,
        fig_title=None,
        ylim=None):

    fig = plt.figure(figsize=fig_size)
    if fig_title:
        plt.title(fig_title)
    ax = fig.gca()

    obs_dim = inputs.shape[-1]
    num_samples = rec_with_forecast.shape[0]

    true_context = inputs[:context_length, :]
    # rec_context = rec_with_forecast[:, :context_length, :]
    true_future = inputs[context_length:, :]
    forecast = rec_with_forecast  # [:, context_length:, :]
    sorted_forecast = np.sort(forecast, axis=0)

    for c in prediction_intervals:
        assert 0.0 <= c <= 100.0

    ps = [50.0] + [
        50.0 + f * c / 2.0
        for c in prediction_intervals
        for f in [-1.0, +1.0]
    ]
    percentiles_sorted = sorted(set(ps))

    def alpha_for_percentile(p):
        return (p / 100.0) ** 0.3

    def quantile(q):
        sample_idx = int(np.round((num_samples - 1) * q))
        return sorted_forecast[sample_idx, :]

    ps_data = [quantile(p / 100.0)
               for p in percentiles_sorted]
    i_p50 = len(percentiles_sorted) // 2

    p50_data = ps_data[i_p50]
    for o in range(obs_dim):
        # True context
        ax.scatter(
            np.arange(context_length), true_context[:, o],
            marker='x',
            color=colors[o],
            s=20)
        # True future
        ax.scatter(
            context_length + np.arange(prediction_length),
            true_future[:, o],
            marker='x',
            color=colors[o],
            s=20)
        # Median forecast
        ax.plot(
            np.arange(context_length + prediction_length),
            p50_data[:, o],
            lw=1,
            color=colors[o])

    if show_mean:
        mean_data = np.mean(sorted_forecast, axis=0)
        for o in range(obs_dim):
            ax.plot(
                context_length + np.arange(prediction_length),
                mean_data[:, o],
                ls=':',
                lw=1,
                color=colors[o])

    for i in range(len(percentiles_sorted) // 2):
        ptile = percentiles_sorted[i]
        alpha = alpha_for_percentile(ptile)
        for o in range(obs_dim):
            ax.fill_between(
                np.arange(context_length + prediction_length),
                ps_data[i][:, o],
                ps_data[-i - 1][:, o],
                facecolor=colors[o],
                alpha=alpha,
                interpolate=True
            )
    ax.set_xlabel('T')
    if ylim is not None:
        ax.set_ylim(ylim)
    return fig


def show_discrete_states(fig_size, discrete_states_lk, segmentation,
                         fig_title=None):
    fig = plt.figure(figsize=fig_size)
    if fig_title:
        plt.title(fig_title)

    ax = fig.add_subplot(1, 1, 1)
    s_seq = np.squeeze(segmentation)
    turning_loc = np.concatenate(
        ([0], np.where(np.diff(s_seq))[0]+1, [s_seq.size]))
    for i in range(discrete_states_lk.shape[-1]):
        ax.plot(np.reshape(discrete_states_lk[Ellipsis, i], [-1]), c=colors[i])
    for tl in turning_loc:
        ax.axvline(tl, color="k", linewidth=2., linestyle="-.")
    ax.set_ylim(-0.1, 1.1)
    return fig


def show_hidden_states(fig_size, zt, segmentation, fig_title=None):
    fig = plt.figure(figsize=fig_size)
    if fig_title:
        plt.title(fig_title)

    ax = fig.add_subplot(1, 1, 1)
    s_seq = np.squeeze(segmentation)
    turning_loc = np.concatenate(
        ([0], np.where(np.diff(s_seq))[0]+1, [s_seq.size]))
    for i in range(zt.shape[-1]):
        ax.plot(zt[:, i])
    for tl in turning_loc:
        ax.axvline(tl, color="k", linewidth=2., linestyle="-.")
    return fig
