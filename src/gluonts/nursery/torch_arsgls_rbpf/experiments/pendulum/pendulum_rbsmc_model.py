import os
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
import pytorch_lightning as pl

import consts
from experiments.default_lightning_model import (
    DefaultLightningModel,
    DefaultExtractTarget,
)
from data.pendulum3D.pendulum3D_coord_environment import generate_dataset
from data.pendulum3D.pendulum_coord_dataset import PendulumCoordDataset
from data.transforms import time_first_collate_fn

from models.base_rbpf_gls import BaseRBSMCGaussianLinearSystem
from visualization.plot_forecasts import make_val_plots_univariate


def remove_groundtruth(item: dict):
    keys_to_remove = ["position", "velocity", "y_gt"]
    return {k: v for k, v in item.items() if k not in keys_to_remove}


class CastDtype(object):
    def __init__(self, model):
        self.model = model

    def __call__(self, item: dict) -> dict:
        return {
            k: v.to(self.model.dtype) if v.is_floating_point() else v
            for k, v in item.items()
        }


class PendulumModel(DefaultLightningModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def prepare_data(self):
        data_path = os.path.join(
            consts.data_dir, getattr(consts.Datasets, self.dataset_name),
        )
        if not os.path.exists(os.path.join(data_path, "train.npz")):
            generate_dataset(dataset_path=data_path)

    def train_dataloader(self):
        tar_extract_collate_fn = DefaultExtractTarget(
            past_length=self.past_length, prediction_length=None,
        )
        to_model_dtype = CastDtype(model=self)
        train_collate_fn = Compose(
            [
                time_first_collate_fn,
                remove_groundtruth,
                tar_extract_collate_fn,
                to_model_dtype,
            ]
        )
        return DataLoader(
            dataset=PendulumCoordDataset(
                file_path=os.path.join(
                    consts.data_dir, self.dataset_name, "train.npz",
                ),
                n_timesteps=self.past_length,
            ),
            batch_size=self.batch_sizes["train"],
            shuffle=True,
            collate_fn=train_collate_fn,
        )

    def val_dataloader(self):
        tar_extract_collate_fn = DefaultExtractTarget(
            past_length=self.past_length,
            prediction_length=self.prediction_length,
        )
        to_model_dtype = CastDtype(model=self)
        val_collate_fn = Compose(
            [time_first_collate_fn, tar_extract_collate_fn, to_model_dtype],
        )
        return DataLoader(
            dataset=PendulumCoordDataset(
                file_path=os.path.join(
                    consts.data_dir, self.dataset_name, "val.npz",
                ),
                n_timesteps=self.past_length + self.prediction_length,
            ),
            batch_size=self.batch_sizes["val"],
            shuffle=False,
            collate_fn=val_collate_fn,
        )

    def test_dataloader(self):
        tar_extract_collate_fn = DefaultExtractTarget(
            past_length=self.past_length,
            prediction_length=self.prediction_length,
        )
        to_model_dtype = CastDtype(model=self)
        test_collate_fn = Compose(
            [time_first_collate_fn, tar_extract_collate_fn, to_model_dtype],
        )
        return DataLoader(
            dataset=PendulumCoordDataset(
                file_path=os.path.join(
                    consts.data_dir, self.dataset_name, "test.npz",
                ),
                n_timesteps=self.past_length + self.prediction_length,
            ),
            batch_size=self.batch_sizes["test"],
            shuffle=False,
            collate_fn=test_collate_fn,
        )

    def validation_step(self, batch, batch_idx):
        batch_no_gt = remove_groundtruth(batch)
        loss = self.loss(**batch_no_gt)
        result = pl.EvalResult(checkpoint_on=loss)
        result.log("val_loss", loss, prog_bar=True)
        if isinstance(self.ssm, BaseRBSMCGaussianLinearSystem):
            if batch_idx == 0:
                make_val_plots_univariate(
                    model=self,
                    data=batch_no_gt,
                    future_target_groundtruth=batch["y_gt"][self.past_length:],
                    idx_particle=None,
                    n_steps_forecast=self.prediction_length,
                    idxs_ts=[0, 1, 2],
                    show=False,
                    savepath=os.path.join(
                        self.logger.log_dir,
                        "plots",
                        f"forecast_ep{self.current_epoch}",
                    ),
                )
        return result
