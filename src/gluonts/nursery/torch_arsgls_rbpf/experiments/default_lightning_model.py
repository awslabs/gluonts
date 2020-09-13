import os
from shutil import copyfile
import inspect
from typing import Optional, Sequence
import torch
from torch.optim import Adam
from torch.optim.optimizer import Optimizer
from pytorch_lightning import LightningModule
import pytorch_lightning as pl

import consts
from models.base_amortized_gls import BaseAmortizedGaussianLinearSystem
from models.base_rbpf_gls import BaseRBSMCGaussianLinearSystem
from models.base_gls import Prediction, Latents
from inference.smc.resampling import EffectiveSampleSizeResampleCriterion
from experiments.model_component_zoo.input_transforms import (
    InputTransformer,
    NoControlsDummyInputTransformer,
)


class DefaultExtractTarget(object):
    def __init__(
        self, past_length: int, prediction_length: Optional[int] = None,
    ):
        self.past_length = past_length
        self.prediction_length = prediction_length

    def __call__(self, item: dict) -> dict:
        transformed_item = {k: v for k, v in item.items() if k != "y"}
        target = item["y"]
        if self.prediction_length is not None:
            if not len(target) >= self.past_length + self.prediction_length:
                raise Exception(
                    f"len(target)=={len(target)}, "
                    f"past_length=={self.past_length}, "
                    f"prediction_length=={self.prediction_length}"
                )
            transformed_item["past_target"] = target[: self.past_length]
            transformed_item["future_target"] = target[
                self.past_length : self.past_length + self.prediction_length
            ]
        else:
            transformed_item["past_target"] = target

        return transformed_item


class DefaultLightningModel(LightningModule):
    """
    Since our SSMs use the same API, most of the experimentation code
    can be shared as well, e.g. training step, loss, and others.
    Data Loading is quite specific for every experiment though.
    Furtunately, lightning seems to move towards having a separate
    dataloader class.
    """

    def __init__(
        self,
        config,  # TODO: remove this once we log hparams with hydra or similar
        ssm: BaseAmortizedGaussianLinearSystem,
        dataset_name,
        lr,
        weight_decay,
        n_epochs,
        batch_sizes,
        past_length,
        n_particle_train,
        n_particle_eval,
        prediction_length,
        ctrl_transformer: Optional[
            InputTransformer
        ] = NoControlsDummyInputTransformer(),
        tar_transformer: Optional[torch.distributions.AffineTransform] = None,
        n_epochs_no_resampling=0,
        n_epochs_freeze_gls_params=0,
        num_batches_per_epoch=50,
        deterministic_forecast: bool = False,
    ):
        super().__init__()
        self.config = config

        self.ctrl_transformer = ctrl_transformer
        self.tar_transformer = tar_transformer
        self.ssm = ssm
        self.past_length = past_length
        self.prediction_length = prediction_length
        self.batch_sizes = batch_sizes
        self.n_epochs_no_resampling = n_epochs_no_resampling
        self.n_epochs_freeze_gls_params = n_epochs_freeze_gls_params
        self.num_batches_per_epoch = num_batches_per_epoch

        self.lr = lr
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.deterministic_forecast = deterministic_forecast

        assert n_particle_train == self.ssm.n_particle
        self._n_particle_train = n_particle_train
        self._n_particle_eval = n_particle_eval

        self.dataset_name = dataset_name

    def forward(
        self,
        past_target: torch.Tensor,
        feat_static_cat: Optional[torch.Tensor] = None,
        past_seasonal_indicators: Optional[torch.Tensor] = None,
        past_time_feat: Optional[torch.Tensor] = None,
        future_seasonal_indicators: Optional[torch.Tensor] = None,
        future_time_feat: Optional[torch.Tensor] = None,
        n_steps_forecast: int = 0,
        deterministic=False,
    ) -> (Sequence[Prediction], Sequence[Latents]):
        if self.tar_transformer is not None:
            past_target = self.tar_transformer.inv(past_target)
        past_controls = self.ctrl_transformer(
            feat_static_cat=feat_static_cat,
            seasonal_indicators=past_seasonal_indicators,
            time_feat=past_time_feat,
        )
        future_controls = (
            self.ctrl_transformer(
                feat_static_cat=feat_static_cat,
                seasonal_indicators=future_seasonal_indicators,
                time_feat=future_time_feat,
            )
            if future_time_feat is not None
            else None
        )
        predictions_inferred, predictions_forecast = self.ssm.predict(
            n_steps_forecast=n_steps_forecast,
            past_targets=past_target,
            past_controls=past_controls,
            future_controls=future_controls,
            deterministic=deterministic,
        )
        # Post-process Sequence of Prediction objects\
        if self.tar_transformer is not None:
            for t in range(len(predictions_inferred)):
                predictions_inferred[t].emissions = self.tar_transformer(
                    predictions_inferred[t].emissions,
                )
            for t in range(len(predictions_forecast)):
                predictions_forecast[t].emissions = self.tar_transformer(
                    predictions_forecast[t].emissions,
                )
        return predictions_inferred, predictions_forecast

    def loss(
        self,
        past_target: torch.Tensor,
        feat_static_cat: Optional[torch.Tensor] = None,
        past_seasonal_indicators: Optional[torch.Tensor] = None,
        past_time_feat: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        T, B = past_target.shape[:2]
        if self.tar_transformer is not None:
            past_target = self.tar_transformer.inv(past_target)
        past_controls = self.ctrl_transformer(
            feat_static_cat=feat_static_cat,
            seasonal_indicators=past_seasonal_indicators,
            time_feat=past_time_feat,
        )
        loss_samplewise = self.ssm.loss(
            past_targets=past_target,
            past_controls=past_controls,
            **{k: v for k, v in kwargs.items() if not "future" in k},
        )
        loss = loss_samplewise.sum(dim=0) / (T * B)
        return loss

    def configure_optimizers(self):
        param_names_gls = [
            name
            for name in dict(self.named_parameters()).keys()
            if ("gls_base_parameters" in name)
            and (not "link_transformers" in name)
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

        optimizer = Adam(
            params=[
                {"params": params_gls, "lr": self.lr},
                {"params": params_except_gls, "lr": self.lr},
            ],
            betas=(0.9, 0.95),
            amsgrad=False,
            weight_decay=self.weight_decay,
        )

        n_iter_lr_decay_one_oom = max(int(self.n_epochs / 2), 1)
        decay_rate = (1 / 10) ** (1 / n_iter_lr_decay_one_oom)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=decay_rate,
        )
        return [optimizer], [scheduler]

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: Optimizer,
        optimizer_idx: int,
        *args,
        **kwargs,
    ) -> None:
        # warmup only certain parameters (all except GLS) if configured.
        is_warmup = epoch < self.n_epochs_freeze_gls_params
        if is_warmup:
            lr_gls = optimizer.param_groups[0]["lr"]
            optimizer.param_groups[0]["lr"] = 0
            optimizer_output = super().optimizer_step(
                epoch, batch_idx, optimizer, optimizer_idx, *args, **kwargs,
            )
            optimizer.param_groups[0]["lr"] = lr_gls
        else:
            optimizer_output = super().optimizer_step(
                epoch, batch_idx, optimizer, optimizer_idx, *args, **kwargs,
            )
        return optimizer_output

    def training_step(self, batch, batch_idx):
        loss = self.loss(**batch)
        result = pl.TrainResult(loss)
        result.log("train_loss", loss)
        return result

    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()
        self.ssm.n_particle = self._n_particle_train

        # Set to no re-sampling if configured for n_epochs_no_resampling epochs.
        if isinstance(self.ssm, BaseRBSMCGaussianLinearSystem):
            if self.n_epochs_no_resampling > 0:
                if self.current_epoch == 0:
                    self._resampling_criterion_fn = (
                        self.ssm.resampling_criterion_fn
                    )
                    self.ssm.resampling_criterion_fn = EffectiveSampleSizeResampleCriterion(
                        min_ess_ratio=0.0,
                    )
                elif self.current_epoch == self.n_epochs_no_resampling:
                    self.ssm.resampling_criterion_fn = (
                        self._resampling_criterion_fn
                    )

    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()
        self.ssm.n_particle = self._n_particle_eval

    def on_test_epoch_start(self) -> None:
        super().on_test_epoch_start()
        self.ssm.n_particle = self._n_particle_eval

    def on_fit_start(self):
        for folder in ["plots", "metrics"]:
            os.makedirs(
                os.path.join(self.logger.log_dir, folder), exist_ok=True,
            )
        # copy config.py and consts.py files. # TODO: Replace this with hydra.
        copyfile(
            src=os.path.abspath(inspect.getfile(self.config.__class__)),
            dst=os.path.join(self.logger.log_dir, "config.py"),
        )
        copyfile(
            src=os.path.abspath(consts.__file__),
            dst=os.path.join(self.logger.log_dir, "consts.py"),
        )
