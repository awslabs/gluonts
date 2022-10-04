# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

from typing import NamedTuple, Optional, Iterable, Dict, Any
import logging

import numpy as np
import pytorch_lightning as pl
import torch.nn as nn

from gluonts.core.component import validated
from gluonts.dataset.common import Dataset
from gluonts.env import env
from gluonts.itertools import Cached
from gluonts.model import Estimator, Predictor
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.transform import Transformation

logger = logging.getLogger(__name__)


class TrainOutput(NamedTuple):
    transformation: Transformation
    trained_net: nn.Module
    trainer: pl.Trainer
    predictor: PyTorchPredictor


class PyTorchLightningEstimator(Estimator):
    """
    An `Estimator` type with utilities for creating PyTorch-Lightning-based
    models.

    To extend this class, one needs to implement three methods:
    `create_transformation`, `create_training_network`, `create_predictor`,
    `create_training_data_loader`, and `create_validation_data_loader`.
    """

    @validated()
    def __init__(
        self,
        trainer_kwargs: Dict[str, Any],
        lead_time: int = 0,
    ) -> None:
        super().__init__(lead_time=lead_time)
        self.trainer_kwargs = trainer_kwargs

    def create_transformation(self) -> Transformation:
        """
        Create and return the transformation needed for training and inference.

        Returns
        -------
        Transformation
            The transformation that will be applied entry-wise to datasets,
            at training and inference time.
        """
        raise NotImplementedError

    def create_lightning_module(self) -> pl.LightningModule:
        """
        Create and return the network used for training (i.e., computing the
        loss).

        Returns
        -------
        pl.LightningModule
            The network that computes the loss given input data.
        """
        raise NotImplementedError

    def create_predictor(
        self,
        transformation: Transformation,
        module,
    ) -> PyTorchPredictor:
        """
        Create and return a predictor object.

        Parameters
        ----------
        transformation
            Transformation to be applied to data before it goes into the model.
        module
            A trained `pl.LightningModule` object.

        Returns
        -------
        Predictor
            A predictor wrapping a `nn.Module` used for inference.
        """
        raise NotImplementedError

    def create_training_data_loader(
        self, data: Dataset, module, **kwargs
    ) -> Iterable:
        """
        Create a data loader for training purposes.

        Parameters
        ----------
        data
            Dataset from which to create the data loader.
        module
            The `pl.LightningModule` object that will receive the batches from
            the data loader.

        Returns
        -------
        Iterable
            The data loader, i.e. and iterable over batches of data.
        """
        raise NotImplementedError

    def create_validation_data_loader(
        self, data: Dataset, module, **kwargs
    ) -> Iterable:
        """
        Create a data loader for validation purposes.

        Parameters
        ----------
        data
            Dataset from which to create the data loader.
        module
            The `pl.LightningModule` object that will receive the batches from
            the data loader.

        Returns
        -------
        Iterable
            The data loader, i.e. and iterable over batches of data.
        """
        raise NotImplementedError

    def train_model(
        self,
        training_data: Dataset,
        validation_data: Optional[Dataset] = None,
        from_predictor: Optional[PyTorchPredictor] = None,
        num_workers: int = 0,
        shuffle_buffer_length: Optional[int] = None,
        cache_data: bool = False,
        ckpt_path: Optional[str] = None,
        **kwargs,
    ) -> TrainOutput:
        transformation = self.create_transformation()

        with env._let(max_idle_transforms=max(len(training_data), 100)):
            transformed_training_data = transformation.apply(
                training_data, is_train=True
            )
            if cache_data:
                transformed_training_data = Cached(transformed_training_data)

            training_network = self.create_lightning_module()

            training_data_loader = self.create_training_data_loader(
                transformed_training_data,
                training_network,
                num_workers=num_workers,
                shuffle_buffer_length=shuffle_buffer_length,
            )

        validation_data_loader = None

        with env._let(max_idle_transforms=max(len(training_data), 100)):
            if validation_data is not None:
                transformed_validation_data = transformation.apply(
                    validation_data, is_train=True
                )
                if cache_data:
                    transformed_validation_data = Cached(
                        transformed_validation_data
                    )

                validation_data_loader = self.create_validation_data_loader(
                    transformed_validation_data,
                    training_network,
                    num_workers=num_workers,
                )

        training_network = self.create_lightning_module()

        if from_predictor is not None:
            training_network.load_state_dict(
                from_predictor.network.state_dict()
            )

        monitor = "train_loss" if validation_data is None else "val_loss"
        checkpoint = pl.callbacks.ModelCheckpoint(
            monitor=monitor, mode="min", verbose=True
        )

        custom_callbacks = self.trainer_kwargs.get("callbacks", [])
        callbacks = [checkpoint] + custom_callbacks
        trainer_kwargs = {**self.trainer_kwargs, "callbacks": callbacks}
        trainer = pl.Trainer(**trainer_kwargs)

        trainer.fit(
            model=training_network,
            train_dataloaders=training_data_loader,
            val_dataloaders=validation_data_loader,
            ckpt_path=ckpt_path,
        )

        logger.info(f"Loading best model from {checkpoint.best_model_path}")
        best_model = training_network.load_from_checkpoint(
            checkpoint.best_model_path
        )

        return TrainOutput(
            transformation=transformation,
            trained_net=best_model,
            trainer=trainer,
            predictor=self.create_predictor(transformation, best_model),
        )

    @staticmethod
    def _worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    def train(
        self,
        training_data: Dataset,
        validation_data: Optional[Dataset] = None,
        num_workers: int = 0,
        shuffle_buffer_length: Optional[int] = None,
        cache_data: bool = False,
        ckpt_path: Optional[str] = None,
        **kwargs,
    ) -> PyTorchPredictor:
        return self.train_model(
            training_data,
            validation_data,
            num_workers=num_workers,
            shuffle_buffer_length=shuffle_buffer_length,
            cache_data=cache_data,
            ckpt_path=ckpt_path,
        ).predictor

    def train_from(
        self,
        predictor: Predictor,
        training_data: Dataset,
        validation_data: Optional[Dataset] = None,
        num_workers: int = 0,
        shuffle_buffer_length: Optional[int] = None,
        cache_data: bool = False,
        ckpt_path: Optional[str] = None,
    ) -> PyTorchPredictor:
        assert isinstance(predictor, PyTorchPredictor)
        return self.train_model(
            training_data,
            validation_data,
            from_predictor=predictor,
            num_workers=num_workers,
            shuffle_buffer_length=shuffle_buffer_length,
            cache_data=cache_data,
            ckpt_path=ckpt_path,
        ).predictor
