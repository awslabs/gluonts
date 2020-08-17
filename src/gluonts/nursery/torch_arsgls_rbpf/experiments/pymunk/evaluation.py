import os
import matplotlib.pyplot as plt
import torch
from box import Box
import ot
import ot.plot
import numpy as np
import argparse
from utils.utils import prepare_logging
import importlib
from torch.utils.data import DataLoader
import consts
from data.transforms import time_first_collate_fn
from data.pymunk_kvae.pymunk_dataset import PymunkDataset
from models.base_rbpf_gls import BaseRBSMCGaussianLinearSystem
from models.kvae import KalmanVariationalAutoEncoder
from inference.smc.normalize import normalize_log_weights


def all_to_numpy(*args):
    return tuple(arg.detach().cpu().numpy() for arg in args)


def compute_wasserstein_distance(img_gt, img_model, metric="euclidean"):
    assert img_gt.ndim == img_model.ndim == 2

    # get positions in x-y-plane for pixels that take value "1" (interpreted as our samples).
    pos_gt = np.stack(np.where(img_gt == 1), axis=-1)
    pos_model = np.stack(np.where(img_model == 1), axis=-1)

    # assume that the binary distribution over pixel value taking value 1 at x-y position
    # is the uniform empirical distribution.
    prob_gt = ot.unif(len(pos_gt))
    prob_model = ot.unif(len(pos_model))

    # euclidean distance times number of pixels --> *total* (not avg) number of pixel movements.
    M = ot.dist(pos_gt, pos_model, metric=metric)
    dist_avg = ot.emd2(prob_gt, prob_model, M)
    # dist_total = dist_avg * len(pos_gt)
    return dist_avg


def compute_metrics(model, batch):
    future_target = batch["future_target"]
    past_target = batch["past_target"]
    batch = {
        k: v for k, v in batch.items() if k != "future_target"
    }  # dont pop

    metrics = Box()
    acc_fcst_rand, acc_filt_rand, acc_all_rand = [], [], []
    acc_fcst_det, acc_filt_det, acc_all_det = [], [], []
    w_dists_rand, w_dists_det = [], []
    for deterministic in [False]:
        past_target_with_particle_dim = past_target.unsqueeze(dim=1)
        future_target_with_particle_dim = future_target.unsqueeze(dim=1)
        if not deterministic:
            past_target_with_particle_dim = past_target_with_particle_dim.repeat(
                [1, model.ssm.n_particle, 1, 1],
            )
            future_target_with_particle_dim = future_target_with_particle_dim.repeat(
                [1, model.ssm.n_particle, 1, 1],
            )
        all_target_with_particle_dim = torch.cat(
            [past_target_with_particle_dim, future_target_with_particle_dim],
        )

        filtered_predictions, forecasted_predictions = model(
            **batch,
            deterministic=deterministic,
            n_steps_forecast=len(future_target),
        )
        filtered_emissions = torch.stack(
            [f.emissions for f in filtered_predictions],
        )
        forecasted_emissions = torch.stack(
            [f.emissions for f in forecasted_predictions],
        )
        all_emissions = torch.cat([filtered_emissions, forecasted_emissions],)

        # Convert all to numpy
        (
            filtered_emissions,
            forecasted_emissions,
            all_emissions,
            past_target_with_particle_dim,
            future_target_with_particle_dim,
            all_target_with_particle_dim,
        ) = all_to_numpy(
            filtered_emissions,
            forecasted_emissions,
            all_emissions,
            past_target_with_particle_dim,
            future_target_with_particle_dim,
            all_target_with_particle_dim,
        )

        # Compute Pixel Accuracy
        # acc_filt = (
        #     np.equal(filtered_emissions, past_target_with_particle_dim)
        #         .astype(np.float64)
        #         .mean(axis=-1)
        # )
        #
        # acc_fcst = (
        #     np.equal(forecasted_emissions, future_target_with_particle_dim)
        #         .astype(np.float64)
        #         .mean(axis=-1)
        # )
        acc = (
            np.equal(all_emissions, all_target_with_particle_dim)
            .astype(np.float64)
            .mean(axis=-1)
        )

        if deterministic:
            # acc_fcst_det.append(acc_fcst)
            # acc_filt_det.append(acc_filt)
            acc_all_det.append(acc)
        else:
            # acc_fcst_rand.append(acc_fcst)
            # acc_filt_rand.append(acc_filt)
            acc_all_rand.append(acc)

        # Compute Wasserstein Distance
        assert all_emissions.shape == all_target_with_particle_dim.shape
        assert (
            all_emissions.shape[0]
            == model.past_length + model.prediction_length
        )
        assert all_emissions.shape[1] == model.ssm.n_particle
        assert all_emissions.shape[3] == 32 * 32
        T = all_emissions.shape[0]
        B = all_emissions.shape[2]
        P = model.ssm.n_particle if not deterministic else 1

        w_dists = np.zeros([T, P, B])
        for t in range(T):
            for p in range(P):
                for b in range(B):
                    w_dists[t, p, b] = compute_wasserstein_distance(
                        img_gt=all_target_with_particle_dim[t, p, b].reshape(
                            [32, 32]
                        ),
                        img_model=all_emissions[t, p, b].reshape([32, 32]),
                    )
        if deterministic:
            w_dists_det.append(w_dists)
        else:
            w_dists_rand.append(w_dists)

    # stack on batch dim and mean
    metrics.acc_rand = np.concatenate(acc_all_rand, axis=-1)
    # metrics.acc_fcst_rand = np.concatenate(acc_fcst_rand, axis=-1)
    # # .mean(axis=(1, 2))
    # metrics.acc_filt_rand = np.concatenate(acc_filt_rand, axis=-1)
    # # .mean(axis=(1, 2))

    # metrics.acc_fcst_det = np.concatenate(acc_fcst_det, axis=-1).mean(axis=[1, 2])
    # metrics.acc_filt_det = np.concatenate(acc_filt_det, axis=-1).mean(axis=[1, 2])
    # mean, std among particle axis. Always mean over data.
    metrics.wasserstein_rand = np.concatenate(
        w_dists_rand, axis=-1
    )  # .mean(axis=1).mean(axis=-1)
    # metrics.w_dists_rand_std = (
    #     np.concatenate(w_dists_rand, axis=-1).std(axis=1).mean(axis=-1)
    # )
    # metrics.w_dists_rand_var = (
    #     np.concatenate(w_dists_rand, axis=-1).var(axis=1).mean(axis=-1)
    # )
    return metrics


def load_model_config(log_paths):
    spec = importlib.util.spec_from_file_location(
        "config", os.path.join(log_paths.root, "config.py")
    )
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-gpu", "--gpu", type=int, default=0)
    parser.add_argument("-run_nr", type=int, default=0)
    parser.add_argument(
        "-experiment_root_log_path",
        type=str,
        default="/home/ubuntu/logs/box/kvae",
        help="path to the root folder of the experiment",
    )
    parser.add_argument("-epoch", type=str, default="final")
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"
    dtype = torch.float64

    log_paths = prepare_logging(
        config=None,
        consts=None,
        copy_config_file=False,
        root_log_path=args.experiment_root_log_path,
    )
    dataset_name = log_paths.root.split("/")[-3]
    config_file = load_model_config(log_paths=log_paths)
    base_config = config_file.base_config

    if base_config.experiment_name == "kvae":
        config = config_file.kvae_config
        make_model = config_file.make_kvae
    elif base_config.experiment_name == "arsgls":
        config = config_file.asgls_config
        make_model = config_file.make_asgls
    else:
        raise ValueError()

    print(f"model: {config.experiment_name}")
    print(f"Using device: {device}; dtype: {dtype}")
    print(f"dataset: {config.dataset_name}")

    dataloaders = {
        data_subset_name: DataLoader(
            dataset=PymunkDataset(
                file_path=os.path.join(
                    consts.data_dir,
                    config.dataset_name,
                    f"{data_subset_name}.npz",
                )
            ),
            batch_size=config.dims.batch
            if data_subset_name == "train"
            else config.batch_size_eval,
            shuffle=True if data_subset_name == "train" else False,
            num_workers=0,
            collate_fn=time_first_collate_fn,
        )
        for data_subset_name in ["train", "test"]
    }

    model = make_model(config=config)
    epoch = args.epoch if args.epoch is not None else "best"

    print(f"loading model from epoch: {epoch}")
    state_dict = torch.load(
        os.path.join(
            # TODO: bad. log_paths messed up. 1 hierarchy (run) missing.
            log_paths.root,
            str(args.run_nr),
            log_paths.model.split("/")[-1],
            f"{epoch}.pt",
        )
    )
    model.load_state_dict(state_dict)

    metrics = compute_metrics(
        model=model,
        model_name=config.experiment_name,
        test_loader=dataloaders["test"],
        config=config,
    )
    np.savez(os.path.join(log_paths.metrics, f"metrics_{epoch}.npz"), metrics)
    print(metrics)


def plot_pymunk_results(
    model,
    batch,
    plot_path,
    deterministic,
    max_data_plot=5,
    max_imgs_per_row=20,
):
    def make_name_extension(deterministic, idx_data, groundtruth=None):
        ext = f"{'det' if deterministic else 'rand'}_B{idx_data}"
        if groundtruth is not None:
            ext = ext + f"_{'groundtruth' if groundtruth else 'model'}"
        return ext

    def stack_images(imgs, horizontal):
        img_list = []
        for im in imgs:
            img_list.append(im)
            img_list.append(
                np.ones_like(im)[:, :3] if horizontal else np.ones_like(im)[:3]
            )
        img_list.pop()  # last no border not needed
        horizontal_stacked_img = np.concatenate(
            img_list, axis=1 if horizontal else 0
        )
        return horizontal_stacked_img

    def plot_imgs_over_time(img, save_path):
        fig, ax = plt.subplots(1, 1)
        ax.axis("off")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.imshow(img, cmap="gray")
        plt.subplots_adjust(wspace=0.025, hspace=0.025)
        plt.tight_layout()
        fig.set_size_inches(
            1.0 * img.shape[1] / img.shape[0], 1, forward=False
        )
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.025)
        plt.close(fig)

    # extract only as much data as needed for the plots.
    n_data_plot = min(batch["past_target"].shape[1], max_data_plot)

    batch = {k: v[:, :n_data_plot] for k, v in batch.items()}
    future_target = batch["future_target"]
    past_target = batch["past_target"]
    batch = {k: v for k, v in batch.items() if k != "future_target"}

    for idx_data in range(n_data_plot):
        os.makedirs(
            os.path.join(plot_path, f"B{idx_data}"), exist_ok=True,
        )

    filtered_predictions, forecasted_predictions = model(
        **batch,
        deterministic=deterministic,
        n_steps_forecast=len(future_target),
    )

    z_filter = torch.stack(
        [fp.latents.variables.auxiliary for fp in filtered_predictions],
    )
    z_forecast = torch.stack(
        [fp.latents.variables.auxiliary for fp in forecasted_predictions],
    )

    y_filter = torch.stack([fp.emissions for fp in filtered_predictions],)
    y_forecast = torch.stack([fp.emissions for fp in forecasted_predictions],)

    if isinstance(model.ssm, BaseRBSMCGaussianLinearSystem):
        log_weights_filter = torch.stack(
            [fp.latents.log_weights for fp in filtered_predictions],
        )
        log_weights_forecast = torch.stack(
            [fp.latents.log_weights for fp in forecasted_predictions],
        )
    elif isinstance(model.ssm, KalmanVariationalAutoEncoder):
        log_weights_filter = torch.zeros_like(z_filter[..., 0])
        log_weights_forecast = torch.zeros_like(z_forecast[..., 0])
    else:
        raise Exception("unknown model class... does it have log-weights?")
    log_norm_weights_filter = normalize_log_weights(log_weights_filter)
    log_norm_weights_forecast = normalize_log_weights(log_weights_forecast)

    z_trajectory = torch.cat([z_filter, z_forecast], dim=0)
    y_trajectory = torch.cat([y_filter, y_forecast], dim=0)
    log_norm_weights_trajectory = torch.cat(
        [log_norm_weights_filter, log_norm_weights_forecast], dim=0,
    )

    # Convert all used tensors to numpy for plotting
    (
        z_filter,
        z_forecast,
        z_trajectory,
        y_filter,
        y_forecast,
        y_trajectory,
        log_weights_filter,
        log_norm_weights_forecast,
        log_norm_weights_trajectory,
        past_target,
        future_target,
    ) = all_to_numpy(
        z_filter,
        z_forecast,
        z_trajectory,
        y_filter,
        y_forecast,
        y_trajectory,
        log_weights_filter,
        log_norm_weights_forecast,
        log_norm_weights_trajectory,
        past_target,
        future_target,
    )

    assert (
        len(z_filter)
        == len(y_filter)
        == len(log_norm_weights_filter)
        == model.past_length
    )
    assert (
        len(z_forecast)
        == len(y_forecast)
        == len(log_norm_weights_forecast)
        == model.prediction_length
    )
    assert (
        len(z_trajectory)
        == len(y_trajectory)
        == len(log_norm_weights_trajectory)
        == model.past_length + model.prediction_length
    )

    if deterministic:
        assert (
            y_trajectory.shape[1:3]
            == y_filter.shape[1:3]
            == y_forecast.shape[1:3]
            == z_trajectory.shape[1:3]
            == (1, n_data_plot,)
        )
    else:
        assert (
            y_trajectory.shape[1:3]
            == y_filter.shape[1:3]
            == y_forecast.shape[1:3]
            == z_trajectory.shape[1:3]
            == (model.ssm.n_particle, n_data_plot,)
        )

    # ********** Plots **********
    t_trajectory = np.arange(len(y_trajectory))
    t_filter = np.arange(len(y_trajectory))
    t_forecast = np.arange(len(y_filter), len(y_filter) + len(y_trajectory))

    # A) Plot auxiliary variable
    for idx_data in range(n_data_plot):
        fig = plt.figure(figsize=[6, 3])
        for idx_dim in range(z_trajectory.shape[-1]):
            if deterministic:
                plt.plot(z_trajectory[:, 0, idx_data, idx_dim])
            else:
                plt.scatter(
                    t_trajectory[:, None].repeat(model.ssm.n_particle, axis=1),
                    z_trajectory[:, :, idx_data, idx_dim],
                    s=(
                        log_norm_weights_trajectory[:, :, idx_data]
                        + np.log(model.ssm.n_particle)
                        + 5
                    ).clip(0, np.inf),
                    alpha=0.5,
                )
        plt.xlabel("t")
        plt.ylabel("z")
        # # for h in lgnd.legendHandles:
        # lgnd = plt.legend(["z1", "z2"])
        # lgnd.legendHandles[0]._sizes = [20]
        # lgnd.legendHandles[1]._sizes = [20]
        plt.axvline(model.past_length, color="black", linestyle="--")
        plt.savefig(
            os.path.join(
                plot_path,
                f"B{idx_data}",
                f"auxiliary_{make_name_extension(deterministic, idx_data)}.pdf",
            )
        )
        plt.close(fig=fig)

    # B) plot filter and forecast one by one.
    n_steps_visual = 20
    n_particles_visual = 20 if not deterministic else 1

    for idx_data in range(n_data_plot):
        img_filt_gt = past_target[:n_steps_visual, idx_data, :].reshape(
            [-1, 32, 32],
        )
        img_fcst_gt = future_target[:n_steps_visual, idx_data, :].reshape(
            [-1, 32, 32],
        )
        for idx_particle in range(n_particles_visual):
            img_filt = y_filter[
                :n_steps_visual, idx_particle, idx_data, :
            ].reshape([-1, 32, 32])
            img_fcst = y_forecast[
                :n_steps_visual, idx_particle, idx_data, :
            ].reshape([-1, 32, 32])
            for is_fcst in [True, False]:
                for is_groundtruth in [True, False]:
                    if is_groundtruth:
                        if idx_particle == 0:
                            imgs = img_fcst_gt if is_fcst else img_filt_gt
                        else:
                            continue
                    else:
                        imgs = img_fcst if is_fcst else img_filt
                    plot_imgs_over_time(
                        img=stack_images(imgs, horizontal=True),
                        save_path=os.path.join(
                            plot_path,
                            f"B{idx_data}",
                            f"{'forecast' if is_fcst else 'filter'}"
                            f"_{make_name_extension(deterministic, idx_data, is_groundtruth)}"
                            f"_P{idx_particle}.pdf",
                        ),
                    )

    # C) Plot groundtruth forecast and model forecast in multi-line.
    idx_particle = 0
    for is_groundtruth in [True, False]:
        rows = int(np.ceil(model.prediction_length / max_imgs_per_row))
        cols = int(np.min([model.prediction_length, max_imgs_per_row]))
        fig, axs = plt.subplots(
            rows, cols, figsize=[cols * 2, rows * 2], squeeze=False
        )
        for idx_time in range(model.prediction_length):
            if is_groundtruth:
                img = future_target[idx_time, idx_data, :].reshape([32, 32],)
            else:
                img = y_forecast[idx_time, idx_particle, idx_data, :].reshape(
                    [32, 32]
                )
            ax = axs[idx_time // cols, idx_time % cols]
            ax.imshow(img, cmap="gray")
            ax.axis("off")
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        plt.subplots_adjust(wspace=0.025, hspace=0.025)
        plt.savefig(
            os.path.join(
                plot_path,
                f"forecast_all_{make_name_extension(deterministic, idx_data, is_groundtruth)}.pdf",
            ),
            bbox_inches="tight",
            pad_inches=0.025,
        )
        plt.close(fig)

    print("done plotting")
