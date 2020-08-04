import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.adam import Adam
from torch.utils.data import DataLoader

from experiments.pymunk.eval_metrics import compute_metrics
import consts
from data.pymunk_kvae.pymunk_dataset import PymunkDataset
from data.transforms import time_first_collate_fn
from experiments.pymunk.configs import (
    base_config,
    kvae_config,
    asgls_config,
    make_asgls,
    make_kvae,
)
from experiments.pymunk.make_filter_forecast import (
    make_kvae_filter_forecast,
    make_arsgls_filter_forecast,
)
from utils.utils import prepare_logging
from copy import deepcopy
import argparse


def train(model, loss_fn, train_loader, log_paths, config, device, dtype):
    model = model.to(dtype).to(device)

    optimizer = Adam(lr=config.lr, params=model.parameters())
    decay_rate = config.lr_decay_rate
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=decay_rate
    )
    optimizer.zero_grad()

    val_data = next(
        iter(dataloaders["train"])
    )  # we did not make validation set...
    val_data = {
        name: val.to(device).to(dtype) if val is not None else None
        for name, val in val_data.items()
    }
    torch.save(
        model.state_dict(), os.path.join(log_paths.model, f"initial.pt")
    )
    loss_best = np.inf
    for idx_epoch in range(config.n_epochs):
        torch.save(
            model.state_dict(),
            os.path.join(log_paths.model, f"{idx_epoch}.pt"),
        )
        if idx_epoch > 0 and idx_epoch % config.decay_steps == 0:
            scheduler.step()

        with torch.no_grad():
            loss_val = loss_fn(model, val_data) / np.prod(
                val_data["y"].shape[:2]
            )
            loss_val = loss_val.detach().cpu().numpy()
        print(f"epoch: {idx_epoch}; loss {loss_val:.3f}")
        if loss_val < loss_best:
            loss_best = loss_val
            torch.save(
                model.state_dict(), os.path.join(log_paths.model, f"best.pt")
            )

        for idx_batch, batch in enumerate(train_loader):
            optimizer.zero_grad()
            batch = {
                name: val.to(device).to(dtype) if val is not None else None
                for name, val in batch.items()
            }
            loss = loss_fn(model, batch) / np.prod(val_data["y"].shape[:2])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.grad_clip_norm
            )
            optimizer.step()

    torch.save(model.state_dict(), os.path.join(log_paths.model, f"final.pt"))
    return model


def plot_results(
    model,
    model_name,
    test_loader,
    log_paths,
    config,
    deterministic,
    n_particle=100,
    n_data=5,
    max_imgs_per_row=20,
    device="cpu",
    dtype=torch.float64,
):
    model = deepcopy(model).to(device).to(dtype)
    model.n_particle = n_particle

    def all_to_numpy(*args):
        return tuple(arg.detach().cpu().numpy() for arg in args)

    if model_name == "kvae":
        make_filter_forecast = make_kvae_filter_forecast
    elif model_name == "arsgls":
        make_filter_forecast = make_arsgls_filter_forecast
    else:
        raise Exception()

    n_loops = int(np.ceil(n_data / config.batch_size_eval))
    (
        z_filter,
        z_forecast,
        y_filter,
        y_forecast,
        log_norm_weights_filter,
        log_norm_weights_forecast,
    ) = ([], [], [], [], [], [])
    test_data_filter, test_data_forecast = {"y": []}, {"y": []}
    for idx_test_data, test_data in enumerate(test_loader):
        _test_data = {
            key: val.to(device).to(dtype)[:, :]
            for key, val in test_data.items()
        }
        if (idx_test_data == n_loops - 1) and (
            n_data % config.batch_size_eval
        ) != 0:
            _test_data = {
                key: val[:, : (n_data % config.batch_size_eval)]
                for key, val in _test_data.items()
            }
        if idx_test_data >= n_loops:
            break

        _test_data_filter = {"y": _test_data["y"][: config.dims.timesteps]}
        _test_data_forecast = {
            "y": _test_data["y"][
                config.dims.timesteps : (
                    config.dims.timesteps + config.prediction_length
                )
            ]
        }

        (
            _z_filter,
            _z_forecast,
            _y_filter,
            _y_forecast,
            _log_norm_weights_filter,
            _log_norm_weights_forecast,
        ) = all_to_numpy(
            *make_filter_forecast(
                model=model,
                data_filter=_test_data_filter,
                prediction_length=config.prediction_length,
                n_particle=n_particle,
                deterministic=deterministic,
            )
        )
        z_filter.append(_z_filter)
        z_forecast.append(_z_forecast)
        y_filter.append(_y_filter)
        y_forecast.append(_y_forecast)
        log_norm_weights_filter.append(_log_norm_weights_filter)
        log_norm_weights_forecast.append(_log_norm_weights_forecast)
        test_data_filter["y"].append(_test_data_filter["y"])
        test_data_forecast["y"].append(_test_data_forecast["y"])

    # concatenate: batch dim is 2 for filter outputs (where we got particle). and 1 for data.
    z_filter = np.concatenate(z_filter, axis=2)
    z_forecast = np.concatenate(z_forecast, axis=2)
    y_filter = np.concatenate(y_filter, axis=2)
    y_forecast = np.concatenate(y_forecast, axis=2)
    log_norm_weights_filter = np.concatenate(log_norm_weights_filter, axis=2)
    log_norm_weights_forecast = np.concatenate(
        log_norm_weights_forecast, axis=2
    )
    test_data_filter = {
        "y": torch.cat(test_data_filter["y"], dim=1).cpu().numpy()
    }
    test_data_forecast = {
        "y": torch.cat(test_data_forecast["y"], dim=1).cpu().numpy()
    }

    z_trajectory = np.concatenate([z_filter, z_forecast], axis=0)
    y_trajectory = np.concatenate([y_filter, y_forecast], axis=0)
    log_norm_weights_trajectory = np.concatenate(
        [log_norm_weights_filter, log_norm_weights_forecast], axis=0
    )

    assert (
        len(z_filter)
        == len(y_filter)
        == len(log_norm_weights_filter)
        == config.dims.timesteps
    )
    assert (
        len(z_forecast)
        == len(y_forecast)
        == len(log_norm_weights_forecast)
        == config.prediction_length
    )
    assert (
        len(z_trajectory)
        == len(y_trajectory)
        == len(log_norm_weights_trajectory)
        == config.dims.timesteps + config.prediction_length
    )

    if deterministic:
        assert (
            y_trajectory.shape[1:3]
            == y_filter.shape[1:3]
            == y_forecast.shape[1:3]
            == z_trajectory.shape[1:3]
            == (1, n_data,)
        )
    else:
        assert (
            y_trajectory.shape[1:3]
            == y_filter.shape[1:3]
            == y_forecast.shape[1:3]
            == z_trajectory.shape[1:3]
            == (n_particle, n_data,)
        )

    # ********** Plots **********
    for idx_data in range(n_data):
        os.makedirs(
            os.path.join(log_paths.plot, f"B{idx_data}"), exist_ok=True
        )

    def make_name_extension(deterministic, idx_data, groundtruth=None):
        ext = f"{'det' if deterministic else 'rand'}_B{idx_data}"
        if groundtruth is not None:
            ext = ext + f"_{'groundtruth' if groundtruth else 'model'}"
        return ext

    t_trajectory = np.arange(len(y_trajectory))
    t_filter = np.arange(len(y_trajectory))
    t_forecast = np.arange(len(y_filter), len(y_filter) + len(y_trajectory))

    # A) Plot auxiliary variable
    for idx_data in range(n_data):
        fig = plt.figure(figsize=[6, 3])
        for idx_dim in range(z_trajectory.shape[-1]):
            if deterministic:
                plt.plot(z_trajectory[:, 0, idx_data, idx_dim])
            else:
                plt.scatter(
                    t_trajectory[:, None].repeat(n_particle, axis=1),
                    z_trajectory[:, :, idx_data, idx_dim],
                    s=(
                        log_norm_weights_trajectory[:, :, idx_data]
                        + np.log(n_particle)
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
        plt.axvline(config.dims.timesteps, color="black", linestyle="--")
        plt.savefig(
            os.path.join(
                log_paths.plot,
                f"B{idx_data}",
                f"auxiliary_{make_name_extension(deterministic, idx_data)}.pdf",
            )
        )
        plt.close(fig=fig)

    # B) plot filter and forecast one by one.
    n_steps_visual = 20
    n_particles_visual = 20 if not deterministic else 1

    def stack_images(imgs, horizontal, border_pixels=3):
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

    for idx_data in range(n_data):
        img_filt_gt = test_data_filter["y"][
            :n_steps_visual, idx_data, :
        ].reshape([-1, 32, 32])
        img_fcst_gt = test_data_forecast["y"][
            :n_steps_visual, idx_data, :
        ].reshape([-1, 32, 32])
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
                            log_paths.plot,
                            f"B{idx_data}",
                            f"{'forecast' if is_fcst else 'filter'}"
                            f"_{make_name_extension(deterministic, idx_data, is_groundtruth)}"
                            f"_P{idx_particle}.pdf",
                        ),
                    )

    # C) Plot groundtruth forecast and model forecast in multi-line.
    idx_particle = 0
    for is_groundtruth in [True, False]:
        rows = int(np.ceil(config.prediction_length / max_imgs_per_row))
        cols = int(np.min([config.prediction_length, max_imgs_per_row]))
        fig, axs = plt.subplots(
            rows, cols, figsize=[cols * 2, rows * 2], squeeze=False
        )
        for idx_time in range(config.prediction_length):
            if is_groundtruth:
                img = test_data_forecast["y"][idx_time, idx_data, :].reshape(
                    [32, 32]
                )
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
                log_paths.plot,
                f"forecast_all_{make_name_extension(deterministic, idx_data, is_groundtruth)}.pdf",
            ),
            bbox_inches="tight",
            pad_inches=0.025,
        )
        plt.close(fig)

    print("done plotting")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('-model_name', default="arsgls", type=str, help="one of {'arsgls', 'kvae'}")
    parser.add_argument("-gpu", "--gpu", type=int, default=0)
    parser.add_argument("-run_nr", type=int, default=0)
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"

    dtype = torch.float64
    seed = args.run_nr
    torch.manual_seed(seed)

    loss_fn_kvae = lambda model, data: model.loss_em(
        **data, rao_blackwellized=kvae_config.rao_blackwellized
    ).sum()
    loss_fn_asgls = lambda model, data: model.loss_forward(**data).sum()
    if base_config.experiment_name == "kvae":
        config = kvae_config
        make_model = make_kvae
        loss_fn = loss_fn_kvae
    elif base_config.experiment_name == "arsgls":
        config = asgls_config
        make_model = make_asgls
        loss_fn = loss_fn_asgls
    else:
        raise ValueError()

    print(f"Using device: {device}; dtype: {dtype}; seed: {seed}")
    print(f"dataset: {config.dataset_name}")
    print(f"model: {config.experiment_name}")

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

    log_paths = prepare_logging(
        config=config, consts=consts, run_nr=args.run_nr
    )
    model = make_model(config=config)

    print(log_paths)
    model = train(
        model=model,
        loss_fn=loss_fn,
        train_loader=dataloaders["train"],
        log_paths=log_paths,
        dtype=dtype,
        device=device,
        config=config,
    )

    print("done training")
    with torch.no_grad():
        plot_results(
            model=model,
            model_name=config.experiment_name,
            test_loader=dataloaders["test"],
            log_paths=log_paths,
            config=config,
            deterministic=False,
            device=device,
        )
        plot_results(
            model=model,
            model_name=config.experiment_name,
            test_loader=dataloaders["test"],
            log_paths=log_paths,
            config=config,
            deterministic=True,
            device=device,
        )
        metrics = compute_metrics(
            model=model,
            model_name=config.experiment_name,
            test_loader=dataloaders["test"],
            config=config,
        )
    np.savez(os.path.join(log_paths.metrics, f"metrics_final.npz"), metrics)
    print(metrics)
