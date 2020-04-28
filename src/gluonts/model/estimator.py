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

# Standard library imports
from typing import NamedTuple, Optional

# Third-party imports
import numpy as np
from mxnet.gluon import HybridBlock
from pydantic import ValidationError

# First-party imports
import gluonts
from gluonts.core import fqname_for
from gluonts.core.component import DType, from_hyperparameters, validated
from gluonts.core.exception import GluonTSHyperparametersError
from gluonts.dataset.common import Dataset
from gluonts.dataset.loader import TrainDataLoader, ValidationDataLoader
from gluonts.model.predictor import Predictor
from gluonts.support.util import get_hybrid_forward_input_names
from gluonts.trainer import Trainer
from gluonts.transform import Transformation


class Estimator:
    """
    An abstract class representing a trainable model.

    The underlying model is trained by calling the `train` method with
    a training `Dataset`, producing a `Predictor` object.
    """

    __version__: str = gluonts.__version__

    prediction_length: int
    freq: str
    lead_time: int

    def __init__(self, lead_time: int = 0, **kwargs) -> None:
        # TODO validation of prediction_length and freq could also
        # TODO be bubbled-up here from subclasses classes
        assert lead_time >= 0, "The value of `lead_time` should be >= 0"

        self.lead_time = lead_time

    def train(
        self, training_data: Dataset, validation_data: Optional[Dataset] = None
    ) -> Predictor:
        """
        Train the estimator on the given data.

        Parameters
        ----------
        training_data
            Dataset to train the model on.
        validation_data
            Dataset to validate the model on during training.

        Returns
        -------
        Predictor
            The predictor containing the trained model.
        """
        raise NotImplementedError

    @classmethod
    def from_hyperparameters(cls, **hyperparameters):
        return from_hyperparameters(cls, **hyperparameters)


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

    @validated()
    def __init__(self, predictor_cls: type, **kwargs) -> None:
        super().__init__(**kwargs)
        self.predictor = predictor_cls(**kwargs)

    def train(
        self,
        training_data: Dataset,
        validation_dataset: Optional[Dataset] = None,
    ) -> Predictor:
        return self.predictor


class TrainOutput(NamedTuple):
    transformation: Transformation
    trained_net: HybridBlock
    predictor: Predictor


class GluonEstimator(Estimator):
    """
    An `Estimator` type with utilities for creating Gluon-based models.

    To extend this class, one needs to implement three methods:
    `create_transformation`, `create_training_network`, `create_predictor`.
    """

    @validated()
    def __init__(
        self, trainer: Trainer, lead_time: int = 0, dtype: DType = np.float32
    ) -> None:
        super().__init__(lead_time=lead_time)
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

    def train_model(
        self,
        training_data: Dataset,
        validation_data: Optional[Dataset] = None,
        num_workers: Optional[int] = None,
        num_prefetch: Optional[int] = None,
        **kwargs,
    ) -> TrainOutput:
        transformation = self.create_transformation()

        transformation.estimate(iter(training_data))

        training_data_loader = TrainDataLoader(
            dataset=training_data,
            transform=transformation,
            batch_size=self.trainer.batch_size,
            num_batches_per_epoch=self.trainer.num_batches_per_epoch,
            ctx=self.trainer.ctx,
            dtype=self.dtype,
            num_workers=num_workers,
            num_prefetch=num_prefetch,
            **kwargs,
        )

        validation_data_loader = None
        if validation_data is not None:
            validation_data_loader = ValidationDataLoader(
                dataset=validation_data,
                transform=transformation,
                batch_size=self.trainer.batch_size,
                ctx=self.trainer.ctx,
                dtype=self.dtype,
                num_workers=num_workers,
                num_prefetch=num_prefetch,
                **kwargs,
            )

        # ensure that the training network is created within the same MXNet
        # context as the one that will be used during training
        with self.trainer.ctx:
            trained_net = self.create_training_network()

        self.trainer(
            net=trained_net,
            input_names=get_hybrid_forward_input_names(trained_net),
            train_iter=training_data_loader,
            validation_iter=validation_data_loader,
        )

        with self.trainer.ctx:
            # ensure that the prediction network is created within the same MXNet
            # context as the one that was used during training
            return TrainOutput(
                transformation=transformation,
                trained_net=trained_net,
                predictor=self.create_predictor(transformation, trained_net),
            )

    def train(
        self,
        training_data: Dataset,
        validation_data: Optional[Dataset] = None,
        num_workers: Optional[int] = None,
        num_prefetch: Optional[int] = None,
        **kwargs,
    ) -> Predictor:
        return self.train_model(
            training_data, validation_data, num_workers, num_prefetch, **kwargs
        ).predictor
