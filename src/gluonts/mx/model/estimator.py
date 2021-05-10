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

from typing import NamedTuple, Optional
from functools import partial

import numpy as np
from mxnet.gluon import HybridBlock
from pydantic import ValidationError

from gluonts.core import fqname_for
from gluonts.core.component import (
    DType,
    from_hyperparameters,
    validated,
    GluonTSHyperparametersError,
)
from gluonts.dataset.common import Dataset
from gluonts.dataset.loader import DataLoader
from gluonts.itertools import Cached
from gluonts.model.estimator import Estimator
from gluonts.model.predictor import Predictor
from gluonts.mx.batchify import as_in_context, batchify
from gluonts.mx.trainer import Trainer
from gluonts.transform import Transformation, TransformedDataset


class TrainOutput(NamedTuple):
    transformation: Transformation
    trained_net: HybridBlock
    predictor: Predictor


class GluonEstimator(Estimator):
    """
    An `Estimator` type with utilities for creating Gluon-based models.

    To extend this class, one needs to implement three methods:
    `create_transformation`, `create_training_network`, `create_predictor`,
    `create_training_data_loader`, and `create_validation_data_loader`.
    """

    @validated()
    def __init__(
        self,
        *,
        trainer: Trainer,
        batch_size: int = 32,
        lead_time: int = 0,
        dtype: DType = np.float32,
    ) -> None:
        super().__init__(lead_time=lead_time)

        assert batch_size > 0, "The value of `batch_size` should be > 0"

        self.batch_size = batch_size
        self.trainer = trainer
        self.dtype = dtype

    @classmethod
    def from_hyperparameters(cls, **hyperparameters) -> "GluonEstimator":
        Model = getattr(cls.__init__, "Model", None)

        if not Model:
            raise AttributeError(
                f"Cannot find attribute Model attached to the "
                f"{fqname_for(cls)}. Most probably you have forgotten to mark "
                f"the class constructor as @validated()."
            )

        try:
            trainer = from_hyperparameters(Trainer, **hyperparameters)

            return cls(
                **Model(**{**hyperparameters, "trainer": trainer}).__dict__
            )
        except ValidationError as e:
            raise GluonTSHyperparametersError from e

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

    def create_training_network(self) -> HybridBlock:
        """
        Create and return the network used for training (i.e., computing the
        loss).

        Returns
        -------
        HybridBlock
            The network that computes the loss given input data.
        """
        raise NotImplementedError

    def create_predictor(
        self, transformation: Transformation, trained_network: HybridBlock
    ) -> Predictor:
        """
        Create and return a predictor object.

        Returns
        -------
        Predictor
            A predictor wrapping a `HybridBlock` used for inference.
        """
        raise NotImplementedError

    def create_training_data_loader(
        self, data: Dataset, **kwargs
    ) -> DataLoader:
        raise NotImplementedError

    def create_validation_data_loader(
        self, data: Dataset, **kwargs
    ) -> DataLoader:
        raise NotImplementedError

    def train_model(
        self,
        training_data: Optional[Dataset] = None,
        validation_data: Optional[Dataset] = None,
        num_workers: Optional[int] = None,
        num_prefetch: Optional[int] = None,
        shuffle_buffer_length: Optional[int] = None,
        cache_data: bool = False,
    ) -> TrainOutput:
        transformation = self.create_transformation()

        transformed_training_data = TransformedDataset(
            training_data, transformation
        )

        training_data_loader = self.create_training_data_loader(
            transformed_training_data
            if not cache_data
            else Cached(transformed_training_data),
            num_workers=num_workers,
            num_prefetch=num_prefetch,
            shuffle_buffer_length=shuffle_buffer_length,
        )

        validation_data_loader = None

        if validation_data is not None:
            transformed_validation_data = TransformedDataset(
                validation_data, transformation
            )

            validation_data_loader = self.create_validation_data_loader(
                transformed_validation_data
                if not cache_data
                else Cached(transformed_validation_data),
            )

        training_network = self.create_training_network()

        self.trainer(
            net=training_network,
            train_iter=training_data_loader,
            validation_iter=validation_data_loader,
        )

        with self.trainer.ctx:
            predictor = self.create_predictor(transformation, training_network)

        return TrainOutput(
            transformation=transformation,
            trained_net=training_network,
            predictor=predictor,
        )

    def train(
        self,
        training_data: Optional[Dataset] = None,
        validation_data: Optional[Dataset] = None,
        num_workers: Optional[int] = None,
        num_prefetch: Optional[int] = None,
        shuffle_buffer_length: Optional[int] = None,
        cache_data: bool = False,
        **kwargs,
    ) -> Predictor:
        return self.train_model(
            training_data=training_data,
            validation_data=validation_data,
            num_workers=num_workers,
            num_prefetch=num_prefetch,
            shuffle_buffer_length=shuffle_buffer_length,
            cache_data=cache_data,
        ).predictor
