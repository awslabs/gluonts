import os
import tqdm
import numpy as np
import mxnet as mx
import torch
from torch.optim.adam import Adam
from torch.utils.data import DataLoader

import consts
from data.pendulum3D.pendulum_coord_dataset import PendulumCoordDataset
from data.transforms import time_first_collate_fn
from experiments.pendulum.config import config, make_model
from utils.utils import prepare_logging
from visualization.plot_forecasts import make_val_plots_pendulum
from inference.smc.resampling import make_criterion_fn_with_ess_threshold


def train(
    log_paths,
    model,
    n_epochs,
    n_epochs_no_resampling,
    lr,
    train_loader,
    gpus,
    dtype=torch.float32,
    n_epochs_until_validate_loss=config.n_epochs_until_validate_loss,
):
    if len(gpus) == 0:
        device = "cpu"
    elif len(gpus) == 1:
        device = f"cuda:{gpus[0]}"
    else:
        device = (
            f"cuda:{gpus[0]}"  # first of the available GPUs to store params on
        )

    print(f"Training on device: {device} [{gpus}]; dtype: {dtype};")
    if len(gpus) > 1:
        model = torch.nn.DataParallel(model, gpus, dim=1)
        m = model.module
    else:
        m = model
    model = model.to(dtype).to(device)

    val_data = next(
        iter(
            DataLoader(  # we did not make validation set...
                dataset=PendulumCoordDataset(
                    file_path=os.path.join(
                        consts.data_dir, dataset_name, "test.npz"
                    ),
                    n_timesteps=dims.timesteps + config.n_steps_forecast,
                ),
                batch_size=500,
                shuffle=False,
                num_workers=0,
                collate_fn=time_first_collate_fn,
            )
        )
    )
    val_data = {
        name: val.to(device).to(dtype) for name, val in val_data.items()
    }

    # ***** Optimizer *****
    n_iter_decay_one_order_of_magnitude = max(int(n_epochs / 2), 1)
    optimizer = Adam(
        params=model.parameters(),
        lr=lr,
        betas=(0.9, 0.95),
        amsgrad=False,
        weight_decay=1e-4,
    )
    decay_rate = (1 / 10) ** (1 / n_iter_decay_one_order_of_magnitude)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=decay_rate
    )
    best_loss = np.inf
    epoch_iter = tqdm.tqdm(range(n_epochs), desc="Epoch", position=0)
    for idx_epoch in epoch_iter:
        # save model
        torch.save(
            model.state_dict(),
            os.path.join(log_paths.model, f"{idx_epoch}.pt"),
        )

        # annealing
        # set re-sampling criterion to ESS = 0.5 after given number of epochs.
        # Start with never re-sampling.
        if idx_epoch == 0:
            print(
                f"\nEpoch: {idx_epoch}: setting resampling criterion fn to ESS = 0.0"
            )
            m.resampling_criterion_fn = make_criterion_fn_with_ess_threshold(
                min_ess_ratio=0.0
            )
        elif idx_epoch == n_epochs_no_resampling:
            print(
                f"\nEpoch: {idx_epoch}: setting resampling criterion fn to ESS = 0.5"
            )
            m.resampling_criterion_fn = make_criterion_fn_with_ess_threshold(
                min_ess_ratio=0.5
            )
        else:
            pass

        # validation: loss
        if idx_epoch % n_epochs_until_validate_loss == 0:
            loss_val = (
                model(y=val_data["y"]).mean(dim=(0, 1)).detach().cpu().numpy()
            )
            for n_step_fcst in [
                config.n_steps_forecast,
                len(val_data["y"]) - 5,
                len(val_data["y"]) - 10,
                len(val_data["y"]) - 20,
            ]:
                for idx_timeseries in range(5):
                    make_val_plots_pendulum(
                        model=model,
                        data={"y": val_data["y"]},
                        y_gt=val_data["y_gt"],
                        idx_particle=None,
                        n_steps_forecast=n_step_fcst,
                        idx_timeseries=idx_timeseries,
                        show=False,
                        savepath=os.path.join(
                            log_paths.plot,
                            f"forecast_b{idx_timeseries}_ep{idx_epoch}_fcst_{n_step_fcst}.pdf",
                        ),
                    )
            print(f"epoch: {idx_epoch}; loss {loss_val:.3f}")
            if loss_val < best_loss:
                best_loss = loss_val
                torch.save(
                    model.state_dict(),
                    os.path.join(log_paths.model, f"best.pt"),
                )

        # training
        for idx_batch, batch in enumerate(train_loader):
            epoch_iter.set_postfix(
                batch=f"{idx_batch}/{len(train_loader)}", refresh=True
            )
            batch = {
                name: val.to(device).to(dtype) for name, val in batch.items()
            }
            optimizer.zero_grad()
            loss = model(batch["y"]).mean(
                dim=(0, 1)
            )  # sum over time and batch
            loss.backward()
            optimizer.step()
        model.trained_epochs = idx_epoch
        scheduler.step()

    print("done training.")
    torch.save(model.state_dict(), os.path.join(log_paths.model, f"final.pt"))
    return model


if __name__ == "__main__":
    dims = config.dims
    log_paths = prepare_logging(config=config, consts=consts)
    print(f"log_paths: \n{log_paths}")

    seed = 45
    mx.random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    dataset_name = "pendulum_3D_coord"
    train_loader = DataLoader(
        dataset=PendulumCoordDataset(
            file_path=os.path.join(consts.data_dir, dataset_name, "train.npz"),
            n_timesteps=dims.timesteps,
        ),
        batch_size=dims.batch,
        shuffle=True,
        num_workers=0,
        collate_fn=time_first_collate_fn,
    )
    test_loader = DataLoader(
        dataset=PendulumCoordDataset(
            file_path=os.path.join(consts.data_dir, dataset_name, "test.npz"),
            n_timesteps=dims.timesteps + config.n_steps_forecast,
        ),
        batch_size=dims.batch,
        shuffle=False,
        num_workers=0,
        collate_fn=time_first_collate_fn,
    )

    model = make_model(config=config)
    model = train(
        log_paths=log_paths,
        model=model,
        n_epochs=config.n_epochs,
        n_epochs_no_resampling=config.n_epochs_no_resampling,
        lr=config.lr,
        train_loader=train_loader,
        gpus=config.gpus,
        dtype=config.dtype,
    )
