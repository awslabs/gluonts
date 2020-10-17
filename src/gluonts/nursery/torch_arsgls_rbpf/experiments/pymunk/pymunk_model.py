from typing import Dict, Union, List
from torch import Tensor
import os
from box import Box
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
import matplotlib.pyplot as plt
import numpy as np

import consts
from experiments.default_lightning_model import (
    DefaultLightningModel,
    DefaultExtractTarget,
)
import data.pymunk_kvae
from data.pymunk_kvae.pymunk_dataset import PymunkDataset
from data.transforms import time_first_collate_fn
from experiments.pymunk.evaluation import (
    compute_metrics,
    plot_pymunk_results,
)
from utils.utils import list_of_dicts_to_dict_of_list


class CastDtype(object):
    def __init__(self, model):
        self.model = model

    def __call__(self, item: dict) -> dict:
        return {
            k: v.to(self.model.dtype) if v.is_floating_point() else v
            for k, v in item.items()
        }


class PymunkModel(DefaultLightningModel):
    def __init__(self, lr_decay_rate, lr_decay_steps,
                 print_validation_metrics=False,
                 *args, **kwargs):
        super().__init__(*args,  **kwargs)
        self.log_metrics = Box()
        self.lr_decay_rate = lr_decay_rate  # custom LR decay rate here.
        self.lr_decay_steps = lr_decay_steps
        self.print_validation_metrics = print_validation_metrics

    def configure_optimizers(self):
        param_names_gls = [
            "ssm.gls_base_parameters._LQinv_logdiag",
            "ssm.gls_base_parameters._LRinv_logdiag",
            "ssm.recurrent_base_parameters._LSinv_logdiag",
            "ssm.state_prior_model.m",
            "ssm.state_prior_model.LVinv_tril",
            "ssm.state_prior_model.LVinv_logdiag",
            "ssm.switch_prior_model.dist.m",
            "ssm.switch_prior_model.dist.LVinv_tril",
            "ssm.switch_prior_model.dist.LVinv_logdiag",
        ]
        params_gls = tuple(
            param
            for name, param in self.named_parameters()
            if name in param_names_gls
        )
        params_except_gls = tuple(
            param
            for name, param in self.named_parameters()
            if name not in param_names_gls
        )
        assert len(params_except_gls) < len(tuple(self.parameters()))
        # assert len(params_gls) == len(param_names_gls)

        optimizer = torch.optim.Adam(
            params=[
                {"params": params_gls, "lr": self.lr},
                {"params": params_except_gls, "lr": self.lr},
            ],
            weight_decay=self.weight_decay,
        )

        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=self.lr_decay_rate,
            ),
            'interval': 'epoch',
            'frequency': self.lr_decay_steps,
        }
        return [optimizer], [scheduler]

    def prepare_data(self):
        data_path = os.path.join(
            consts.data_dir, getattr(consts.Datasets, self.dataset_name),
        )
        if not os.path.exists(os.path.join(data_path, "train.npz")):
            dataset_pkg = getattr(data.pymunk_kvae, self.dataset_name)
            dataset_pkg.generate_dataset()

    def train_dataloader(self):
        tar_extract_collate_fn = DefaultExtractTarget(
            past_length=self.past_length, prediction_length=None,
        )
        to_model_dtype = CastDtype(model=self)
        train_collate_fn = Compose(
            [time_first_collate_fn, tar_extract_collate_fn, to_model_dtype],
        )
        return DataLoader(
            dataset=PymunkDataset(
                file_path=os.path.join(
                    consts.data_dir, self.dataset_name, "train.npz",
                ),
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
        test_collate_fn = Compose(
            [time_first_collate_fn, tar_extract_collate_fn, to_model_dtype],
        )
        return DataLoader(
            dataset=PymunkDataset(
                file_path=os.path.join(
                    consts.data_dir, self.dataset_name, "val.npz",
                ),
            ),
            batch_size=self.batch_sizes["val"],
            shuffle=False,
            collate_fn=test_collate_fn,
        )

    def validation_step(self, batch, batch_idx):
        loss = self.loss(**batch)
        self.log("val_loss", loss, prog_bar=True)
        aggregated_metrics = {"val_loss": loss}

        try:
            # Plot
            if batch_idx == 0 and (self.trainer.current_epoch % 40 == 0) \
                    and (self.trainer.current_epoch > 0):
                os.makedirs(
                    os.path.join(
                        self.logger.log_dir,
                        f"epoch_{self.trainer.current_epoch}",
                        "plots",
                    ),
                    exist_ok=True,
                )
                plot_pymunk_results(
                    model=self,
                    batch=batch,
                    deterministic=False,
                    plot_path=os.path.join(
                        self.logger.log_dir,
                        f"epoch_{self.trainer.current_epoch}",
                        "plots",
                    ),
                )
            if self.trainer.current_epoch % 10 == 0:
                metrics = compute_metrics(
                    model=self,
                    batch=batch,
                    n_last_timesteps_wasserstein=5,
                    n_particles_wasserstein=32,
                )
                for k, v in metrics.items():
                    v_agg = torch.tensor(v, dtype=self.dtype).mean()
                    aggregated_metrics.update({k: v_agg})
        except:
            print("Warning: validation failed. Probably due to FP32 problems?")
    
        for k, v in aggregated_metrics.items():
            self.log(k, v)

        return aggregated_metrics

    def validation_epoch_end(
        self,
        outputs: Union[List[Dict[str, Tensor]], List[List[Dict[str, Tensor]]]],
    ) -> Dict[str, Dict[str, Tensor]]:
        aggregated_metrics = {
            k: sum(v) / len(v)
            for k, v in list_of_dicts_to_dict_of_list(outputs).items()
        }
        for k, v in aggregated_metrics.items():
            self.log(k, v, prog_bar=True)
        if self.print_validation_metrics:
            print(f"epoch: {self.current_epoch}: ", aggregated_metrics)
        return aggregated_metrics

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
            dataset=PymunkDataset(
                file_path=os.path.join(
                    consts.data_dir, self.dataset_name, "test.npz",
                ),
            ),
            batch_size=self.batch_sizes["test"],
            shuffle=False,
            collate_fn=test_collate_fn,
        )

    def test_step(self, batch, batch_idx):
        # 1) Plot
        if batch_idx == 0:
            for deterministic in [True, False]:
                n_particle = self.ssm.n_particle
                if deterministic:
                    self.ssm.n_particle = 1
                plot_pymunk_results(
                    model=self,
                    batch=batch,
                    deterministic=deterministic,
                    plot_path=os.path.join(self.logger.log_dir, "plots"),
                )
                self.ssm.n_particle = n_particle

        # 2) Compute metrics
        metrics = compute_metrics(model=self, batch=batch)
        # non-scalars in numpy. Otherwise lightning complains,
        # because it tries to convert it.
        return metrics

    def test_epoch_end(self, outputs):
        metrics = list_of_dicts_to_dict_of_list(outputs)
        metric_names = [k for k in metrics.keys() if k != "meta"]
        result = {}
        for metric_name in metric_names:
            agg_metrics = {}
            metrics_cat = np.concatenate(metrics[metric_name], axis=-1)

            if metrics_cat.ndim == 3:
                # mean, std, var over particle dim. Always mean over batch/data.
                agg_metrics["mean"] = metrics_cat.mean(axis=1).mean(axis=-1)
                agg_metrics["std"] = metrics_cat.std(axis=1).mean(axis=-1)
                agg_metrics["var"] = metrics_cat.var(axis=1).mean(axis=-1)
                # TODO: currently there is no way to log non-scalar metrics in
                #  pytorch-lightning? This is a temporary hack to make the
                #  metric over time (vector) available to outside for plotting.
                #  But this should abolutely be a feature of lightning...

                self.log_metrics[metric_name] = Box()
                for which_agg, agg_metric in agg_metrics.items():
                    self.log_metrics[metric_name][which_agg] = agg_metric
                    name = f"{metric_name}_{which_agg}"
                    result[name] = agg_metric.mean(axis=0)
                    self.log(name, result[name])
            elif metrics_cat.ndim == 2:
                agg_metrics = metrics_cat.mean(axis=-1)  # Batch
                self.log_metrics[metric_name] = agg_metrics
                result[metric_name] = agg_metrics.mean(axis=0)
                self.log(metric_name, result[metric_name])
            else:
                raise ValueError(
                    f"metric '{metric_name}' has unexpected "
                    f"tensor dims: {metrics_cat.shape}"
                )

        # There are no methods in lightning or even tensorboard to log
        # 1D tensors for a standard line-plot. So we save them with
        # numpy instead and make a standard matplotlib plot.
        np.savez(
            os.path.join(
                self.logger.log_dir, "metrics", "sequence_metrics.npz",
            ),
            self.log_metrics,
        )

        time = np.arange(self.past_length + self.prediction_length)
        for metric_name in metric_names:
            fig = plt.figure()
            if isinstance(self.log_metrics[metric_name], dict):
                m = self.log_metrics[metric_name]["mean"]
                plt.plot(time, m, label="mean")
                # std = self.log_metrics[metric_name]["std"]
                # (lower, upper) = (m - 3 * std, m + 3 * std)
                # plt.fill_between(time, lower, upper, alpha=0.25, label="3 std")
            else:
                plt.plot(
                    time, self.log_metrics[metric_name], label=metric_name
                )

            plt.axvline(
                self.past_length - 1,
                linestyle="--",
                color="black",
                label="_nolegend_",
            )
            plt.legend()
            plt.xlabel("t")
            plt.ylabel(metric_name)
            plt.savefig(
                os.path.join(
                    self.logger.log_dir, "plots", f"{metric_name}.pdf",
                ),
                bbox_inches="tight",
                pad_inches=0.025,
            )
            plt.close(fig)

        return result
