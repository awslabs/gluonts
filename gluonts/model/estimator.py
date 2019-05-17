# Third-party imports
from typing import Tuple

# Standard library imports
import numpy as np
from mxnet.gluon import HybridBlock
from pydantic import ValidationError

# First-party imports
from gluonts.core import fqname_for
from gluonts.core.component import DType, from_hyperparameters, validated
from gluonts.core.exception import GluonTSHyperparametersError
from gluonts.dataset.common import Dataset
from gluonts.dataset.loader import TrainDataLoader
from gluonts.model.predictor import Predictor
from gluonts.support.util import get_hybrid_forward_input_names
from gluonts.trainer import Trainer
from gluonts.transform import Transformation


class Estimator:
    prediction_length: int
    freq: str

    def train(self, training_data: Dataset) -> Predictor:
        raise NotImplementedError


class DummyEstimator(Estimator):
    @validated()
    def __init__(self, predictor_cls: type, **kwargs) -> None:
        self.predictor = predictor_cls(**kwargs)

    def train(self, training_data: Dataset) -> Predictor:
        return self.predictor

    @classmethod
    def from_hyperparameters(cls, **hyperparameters):
        return from_hyperparameters(cls, **hyperparameters)


class GluonEstimator(Estimator):
    """
    An estimator with some utilities for creating Estimators from gluon

    To use this implement these three methods

    - create_transformation
    - create_training_network
    - create_predictor
    """

    @validated()
    def __init__(
        self, trainer: Trainer, float_type: DType = np.float32
    ) -> None:
        self.trainer = trainer
        self.float_type = float_type

    @classmethod
    def from_hyperparameters(cls, **hyperparameters) -> 'GluonEstimator':
        Model = getattr(cls.__init__, 'Model', None)

        if not Model:
            raise AttributeError(
                f'Cannot find attribute Model attached to the '
                f'{fqname_for(cls)}. Most probably you have forgotten to mark '
                f'the class constructor as @validated().'
            )

        try:
            trainer = from_hyperparameters(Trainer, **hyperparameters)
            return cls(
                **Model(**{**hyperparameters, 'trainer': trainer}).__values__
            )
        except ValidationError as e:
            raise GluonTSHyperparametersError from e

    def create_transformation(self) -> Transformation:
        """
        Create and return the transformation needed for training and inference.
        """
        raise NotImplementedError

    def create_training_network(self) -> HybridBlock:
        """
        Create and return the network used for training (i.e., computing the
        loss).
        """
        raise NotImplementedError

    def create_predictor(
        self, transformation: Transformation, trained_network: HybridBlock
    ) -> Predictor:
        """
        Create and return a predictor object.
        """
        raise NotImplementedError

    def train_model(
        self, training_data: Dataset
    ) -> Tuple[Transformation, HybridBlock]:
        transformation = self.create_transformation()

        transformation.estimate(iter(training_data))

        training_data_loader = TrainDataLoader(
            dataset=training_data,
            transform=transformation,
            batch_size=self.trainer.batch_size,
            num_batches_per_epoch=self.trainer.num_batches_per_epoch,
            ctx=self.trainer.ctx,
            float_type=self.float_type,
        )

        # ensure that the training network is created within the same MXNet
        # context as the one that will be used during training
        with self.trainer.ctx:
            trained_net = self.create_training_network()

        self.trainer(
            net=trained_net,
            input_names=get_hybrid_forward_input_names(trained_net),
            train_iter=training_data_loader,
        )

        return transformation, trained_net

    def train(self, training_data: Dataset) -> Predictor:

        training_transformation, trained_net = self.train_model(training_data)

        # ensure that the prediction network is created within the same MXNet
        # context as the one that was used during training
        with self.trainer.ctx:
            return self.create_predictor(training_transformation, trained_net)
