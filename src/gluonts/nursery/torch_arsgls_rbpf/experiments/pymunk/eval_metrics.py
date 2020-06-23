import torch
from box import Box
from experiments.pymunk.make_filter_forecast import make_kvae_filter_forecast, \
    make_arsgls_filter_forecast
from copy import deepcopy
import ot
import ot.plot
import numpy as np
import tqdm
import argparse
from utils.utils import prepare_logging
import importlib
import os
from torch.utils.data import DataLoader
from data.transforms import time_first_collate_fn
import consts
from data.pymunk_kvae.pymunk_dataset import PymunkDataset


def compute_wasserstein_distance(img_gt, img_model, metric='euclidean'):
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


def compute_metrics(model, model_name, test_loader, config, n_particle=100):
    # model = deepcopy(model)
    device = model.state_prior_model.m.device
    dtype = model.state_prior_model.m.dtype
    model.n_particle = n_particle

    def all_to_numpy(*args):
        return tuple(arg.detach().cpu().numpy() for arg in args)

    if model_name == "kvae":
        make_filter_forecast = make_kvae_filter_forecast
    elif model_name == "arsgls":
        make_filter_forecast = make_arsgls_filter_forecast
    else:
        raise Exception()

    metrics = Box()
    acc_fcst_rand, acc_filt_rand, = [], []
    acc_fcst_det, acc_filt_det, = [], []
    w_dists_rand, w_dists_det = [], []
    for deterministic in [False]:
        for idx_test_data, test_data in enumerate(tqdm.tqdm(test_loader)):
            print(
                f"eval test batch: {idx_test_data}/{int(np.ceil(1000 / config.batch_size_test))}")
            # Data and groundtruth
            test_data_filter = {
                "y": test_data['y'][:config.dims.timesteps].to(dtype).to(device)
            }
            test_data_forecast = {
                "y": test_data['y'][
                     config.dims.timesteps: config.dims.timesteps
                                            + config.prediction_length].to(
                    dtype).to(device)}

            y_filter_groundtruth = test_data_filter["y"][:, None, ...].repeat(
                [1, (1 if deterministic else n_particle), 1, 1]).cpu().numpy()
            y_forecast_groundtruth = test_data_forecast["y"][:, None,
                                     ...].repeat(
                [1, (1 if deterministic else n_particle), 1, 1]).cpu().numpy()
            y_trajectory_groundtruth = np.concatenate(
                [y_filter_groundtruth, y_forecast_groundtruth], axis=0)

            # Get filter and forecast results
            z_filter, z_forecast, y_filter, y_forecast, \
            log_norm_weights_filter, log_norm_weights_forecast = all_to_numpy(
                *make_filter_forecast(
                    model=model,
                    data_filter=test_data_filter,
                    prediction_length=config.prediction_length,
                    n_particle=n_particle,
                    deterministic=deterministic,
                ))

            # Compute Pixel Accuracy
            acc_fcst = np.equal(y_forecast, y_forecast_groundtruth).astype(
                np.float64).mean(axis=-1)
            acc_filt = np.equal(y_filter, y_filter_groundtruth).astype(
                np.float64).mean(axis=-1)
            if deterministic:
                acc_fcst_det.append(acc_fcst)
                acc_filt_det.append(acc_filt)
            else:
                acc_fcst_rand.append(acc_fcst)
                acc_filt_rand.append(acc_filt)

            # Compute Wasserstein Distance
            y_trajectory = np.concatenate([y_filter, y_forecast], axis=0)
            assert y_trajectory.shape == y_trajectory_groundtruth.shape
            assert y_trajectory.shape[
                       0] == config.dims.timesteps + config.prediction_length
            assert y_trajectory.shape[1] == n_particle
            assert y_trajectory.shape[3] == 32 * 32

            T = y_trajectory.shape[0]
            B = y_trajectory.shape[2]
            P = n_particle if not deterministic else 1

            w_dists = np.zeros([T, P, B])
            for t in range(T):
                for p in range(P):
                    for b in range(B):
                        w_dists[t, p, b] = compute_wasserstein_distance(
                            img_gt=y_trajectory_groundtruth[t, p, b].reshape(
                                [32, 32]),
                            img_model=y_trajectory[t, p, b].reshape([32, 32]),
                        )
            if deterministic:
                w_dists_det.append(w_dists)
            else:
                w_dists_rand.append(w_dists)

    # stack on batch dim and mean
    metrics.acc_fcst_rand = np.concatenate(acc_fcst_rand, axis=-1).mean(
        axis=(1, 2))
    metrics.acc_filt_rand = np.concatenate(acc_filt_rand, axis=-1).mean(
        axis=(1, 2))
    # metrics.acc_fcst_det = np.concatenate(acc_fcst_det, axis=-1).mean(axis=[1, 2])
    # metrics.acc_filt_det = np.concatenate(acc_filt_det, axis=-1).mean(axis=[1, 2])
    # mean, std among particle axis. Always mean over data.
    metrics.w_dists_rand_mean = np.concatenate(w_dists_rand, axis=-1).mean(
        axis=1).mean(axis=-1)
    metrics.w_dists_rand_std = np.concatenate(w_dists_rand, axis=-1).std(
        axis=1).mean(axis=-1)
    metrics.w_dists_rand_var = np.concatenate(w_dists_rand, axis=-1).var(
        axis=1).mean(axis=-1)
    return metrics


def load_model_config(log_paths):
    spec = importlib.util.spec_from_file_location(
        "config", os.path.join(log_paths.root, "config.py"))
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', '--gpu', type=int, default=0)
    parser.add_argument('-run_nr', type=int, default=0)
    parser.add_argument('-experiment_root_log_path', type=str,
                        default='/home/ubuntu/logs/box/kvae',
                        help='path to the root folder of the experiment')
    parser.add_argument('-epoch', type=str, default="final")
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"
    dtype = torch.float64

    log_paths = prepare_logging(
        config=None, consts=None, copy_config_file=False,
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
            dataset=PymunkDataset(file_path=os.path.join(
                consts.data_dir, config.dataset_name,
                f"{data_subset_name}.npz")),
            batch_size=config.dims.batch if data_subset_name == "train" else config.batch_size_test,
            shuffle=True if data_subset_name == "train" else False,
            num_workers=0,
            collate_fn=time_first_collate_fn,
        ) for data_subset_name in ["train", "test"]}

    model = make_model(config=config)
    epoch = args.epoch if args.epoch is not None else "best"

    print(f"loading model from epoch: {epoch}")
    state_dict = torch.load(
        os.path.join(
            # TODO: bad. log_paths messed up. 1 hierarchy (run) missing.
            log_paths.root, str(args.run_nr), log_paths.model.split("/")[-1],
            f"{epoch}.pt"))
    model.load_state_dict(state_dict)

    metrics = compute_metrics(model=model, model_name=config.experiment_name,
                              test_loader=dataloaders["test"], config=config)
    np.savez(os.path.join(log_paths.metrics, f"metrics_{epoch}.npz"), metrics)
    print(metrics)
