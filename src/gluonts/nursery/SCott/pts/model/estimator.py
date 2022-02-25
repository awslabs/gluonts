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


from abc import ABC, abstractmethod
from typing import List, NamedTuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pts import Trainer
from pts.dataset import Dataset
from pts.dataset.loader import InferenceDataLoader
from pts.dataset.transformed_iterable_dataset import (
    TransformedIterableDataset,
    TransformedGroupedIterableDataset,
    FullBatchDataset,
    FullGroupBatchDataset,
)
from pts.transform import Transformation
from .predictor import Predictor
from .utils import get_module_forward_input_names


class Estimator(ABC):
    prediction_length: int
    freq: str

    @abstractmethod
    def train(self, training_data: Dataset) -> Predictor:
        pass


class DummyEstimator(Estimator):
    """
    An `Estimator` that, upon training, simply returns a pre-constructed
    `Predictor`.

    Parameters
    ----------
    predictor_cls
        `Predictor` class to instantiate.
    **kwargs
        Keyword arguments to pass to the predictor constructor.
    """

    def __init__(self, predictor_cls: type, **kwargs) -> None:
        self.predictor = predictor_cls(**kwargs)

    def train(self, training_data: Dataset) -> Predictor:
        return self.predictor


class TrainOutput(NamedTuple):
    transformation: Transformation
    trained_net: nn.Module
    predictor: Predictor


class PTSEstimator(Estimator):
    def __init__(self, trainer: Trainer, dtype: np.dtype = np.float32) -> None:
        self.trainer = trainer
        self.dtype = dtype

    @abstractmethod
    def create_transformation(self) -> Transformation:
        """
        Create and return the transformation needed for training and inference.

        Returns
        -------
        Transformation
            The transformation that will be applied entry-wise to datasets,
            at training and inference time.
        """
        pass

    @abstractmethod
    def create_training_network(self, device: torch.device) -> nn.Module:
        """
        Create and return the network used for training (i.e., computing the
        loss).

        Returns
        -------
        nn.Module
            The network that computes the loss given input data.
        """
        pass

    @abstractmethod
    def create_predictor(
        self,
        transformation: Transformation,
        trained_network: nn.Module,
        device: torch.device,
    ) -> Predictor:
        """
        Create and return a predictor object.

        Returns
        -------
        Predictor
            A predictor wrapping a `nn.Module` used for inference.
        """
        pass

    def train_model(self, data_package) -> TrainOutput:
        transformation = self.create_transformation()
        transformation_full_batch = self.create_transformation(
            is_full_batch=True
        )

        training_iter_dataset = TransformedIterableDataset(
            dataset=data_package["whole_data"],
            is_train=True,
            transform=transformation,
        )

        training_data_loader = DataLoader(
            training_iter_dataset,
            batch_size=self.trainer.batch_size,
            num_workers=self.trainer.num_workers,
            pin_memory=self.trainer.pin_memory,
        )

        rand_anchor_loader = DataLoader(
            training_iter_dataset,
            batch_size=self.trainer.batch_size,
            num_workers=self.trainer.num_workers,
            pin_memory=self.trainer.pin_memory,
        )

        test_iter_dataset = FullBatchDataset(
            dataset=data_package["val_data"],
            is_train=True,
            transform=transformation_full_batch,
        )

        test_data_loader = DataLoader(
            test_iter_dataset,
            batch_size=8192,
            num_workers=self.trainer.num_workers,
            pin_memory=self.trainer.pin_memory,
        )

        full_batch_dataset = FullBatchDataset(
            dataset=data_package["whole_data"],
            is_train=True,
            transform=transformation_full_batch,
        )

        full_batch_loader = DataLoader(
            full_batch_dataset,
            batch_size=8192,
            num_workers=self.trainer.num_workers,
            pin_memory=self.trainer.pin_memory,
        )

        # ensure that the training network is created on the same device
        trained_net = self.create_training_network(self.trainer.device)

        self.trainer(
            net=trained_net,
            input_names=get_module_forward_input_names(trained_net),
            data_loaders={
                "training_data_loader": training_data_loader,
                "validation_data_loader": test_data_loader,
                "full_batch_loader": full_batch_loader,
                "rand_anchor_loader": rand_anchor_loader,
            },
        )

        return TrainOutput(
            transformation=transformation,
            trained_net=trained_net,
            predictor=self.create_predictor(
                transformation, trained_net, self.trainer.device
            ),
        )

    def train(self, data_package) -> Predictor:
        return self.train_model(data_package).predictor

    def stratified_train_model(self, data_package) -> TrainOutput:
        transformation = self.create_transformation()
        transformation_full_batch = self.create_transformation(
            is_full_batch=True
        )

        anchor_iter_dataset = TransformedGroupedIterableDataset(
            list_of_dataset=data_package["group_data"],
            is_train=True,
            transform=transformation,
            batch_size=self.trainer.batch_size,
        )

        anchor_data_loader = DataLoader(
            anchor_iter_dataset,
            batch_size=self.trainer.batch_size * self.trainer.num_strata,
            num_workers=self.trainer.num_workers,
            pin_memory=self.trainer.pin_memory,
        )

        training_iter_dataset = TransformedIterableDataset(
            dataset=data_package["whole_data"],
            is_train=True,
            transform=transformation,
        )

        training_data_loader = DataLoader(
            training_iter_dataset,
            batch_size=self.trainer.batch_size,
            num_workers=self.trainer.num_workers,
            pin_memory=self.trainer.pin_memory,
        )

        test_iter_dataset = FullBatchDataset(
            dataset=data_package["val_data"],
            is_train=True,
            transform=transformation_full_batch,
        )

        test_data_loader = DataLoader(
            test_iter_dataset,
            batch_size=8192,
            num_workers=self.trainer.num_workers,
            pin_memory=self.trainer.pin_memory,
        )

        full_batch_dataset = FullBatchDataset(
            dataset=data_package["whole_data"],
            is_train=True,
            transform=transformation_full_batch,
        )

        full_batch_loader = DataLoader(
            full_batch_dataset,
            batch_size=8192,
            num_workers=self.trainer.num_workers,
            pin_memory=self.trainer.pin_memory,
        )

        # ensure that the training network is created on the same device
        trained_net = self.create_training_network(self.trainer.device)

        self.trainer(
            net=trained_net,
            input_names=get_module_forward_input_names(trained_net),
            data_loaders={
                "training_data_loader": training_data_loader,
                "validation_data_loader": test_data_loader,
                "anchor_data_loader": anchor_data_loader,
                "full_batch_loader": full_batch_loader,
                "group_ratio": data_package["group_ratio"],
            },
        )

        return TrainOutput(
            transformation=transformation,
            trained_net=trained_net,
            predictor=self.create_predictor(
                transformation, trained_net, self.trainer.device
            ),
        )

    def stratified_train(self, data_package) -> Predictor:
        return self.stratified_train_model(data_package).predictor
