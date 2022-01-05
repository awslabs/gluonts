from typing import List, Optional

import torch
import torch.nn as nn

from pts import Trainer
from pts.dataset import FieldName
from pts.model import PTSEstimator, PTSPredictor, copy_parameters
from pts.modules import DistributionOutput, StudentTOutput, NormalOutput
from pts.transform import (
    Transformation,
    Chain,
    InstanceSplitter,
    ExpectedNumInstanceSampler,
)
from pts.transform.sampler import CustomUniformSampler
from .simple_feedforward_network import (
    SimpleFeedForwardTrainingNetwork,
    SimpleFeedForwardPredictionNetwork,
)


class SimpleFeedForwardEstimator(PTSEstimator):
    """
    SimpleFeedForwardEstimator shows how to build a simple MLP model predicting
    the next target time-steps given the previous ones.

    Given that we want to define a pytorch model trainable by SGD, we inherit the
    parent class `PTSEstimator` that handles most of the logic for fitting a
    neural-network.

    We thus only have to define:

    1. How the data is transformed before being fed to our model::

        def create_transformation(self) -> Transformation

    2. How the training happens::

        def create_training_network(self) -> nn.Module

    3. how the predictions can be made for a batch given a trained network::

        def create_predictor(
             self,
             transformation: Transformation,
             trained_net: nn.Module,
        ) -> Predictor


    Parameters
    ----------
    freq
        Time time granularity of the data
    prediction_length
        Length of the prediction horizon
    trainer
        Trainer object to be used (default: Trainer())
    num_hidden_dimensions
        Number of hidden nodes in each layer (default: [40, 40])
    context_length
        Number of time units that condition the predictions
        (default: None, in which case context_length = prediction_length)
    distr_output
        Distribution to fit (default: StudentTOutput())
    batch_normalization
        Whether to use batch normalization (default: False)
    mean_scaling
        Scale the network input by the data mean and the network output by
        its inverse (default: True)
    num_parallel_samples
        Number of evaluation samples per time series to increase parallelism during inference.
        This is a model optimization that does not affect the accuracy (default: 100)
    """

    def __init__(
        self,
        freq: str,
        prediction_length: int,
        trainer: Trainer = Trainer(),
        num_hidden_dimensions: Optional[List[int]] = None,
        context_length: Optional[int] = None,
        distr_output: DistributionOutput = StudentTOutput(),
        batch_normalization: bool = False,
        mean_scaling: bool = True,
        num_parallel_samples: int = 100,
    ) -> None:
        """
        Defines an estimator. All parameters should be serializable.
        """
        super().__init__(trainer=trainer)

        self.num_hidden_dimensions = (
            num_hidden_dimensions
            if num_hidden_dimensions is not None
            else list([40, 40])
        )
        self.prediction_length = prediction_length
        self.context_length = (
            context_length if context_length is not None else prediction_length
        )
        self.freq = freq
        self.distr_output = distr_output
        self.batch_normalization = batch_normalization
        self.mean_scaling = mean_scaling
        self.num_parallel_samples = num_parallel_samples

    # here we do only a simple operation to convert the input data to a form
    # that can be digested by our model by only splitting the target in two, a
    # conditioning part and a to-predict part, for each training example.
    # For a more complex transformation example, see the `pts.model.deepar`
    # transformation that includes time features, age feature, observed values
    # indicator, etc.
    def create_transformation(self, is_full_batch=False) -> Transformation:
        return Chain(
            [
                InstanceSplitter(
                    target_field=FieldName.TARGET,
                    is_pad_field=FieldName.IS_PAD,
                    start_field=FieldName.START,
                    forecast_start_field=FieldName.FORECAST_START,
                    # train_sampler=ExpectedNumInstanceSampler(num_instances=1),
                    train_sampler=CustomUniformSampler(),
                    past_length=self.context_length,
                    is_full_batch=is_full_batch,
                    future_length=self.prediction_length,
                    time_series_fields=[],  # [FieldName.FEAT_DYNAMIC_REAL]
                )
            ]
        )

    # defines the network, we get to see one batch to initialize it.
    # the network should return at least one tensor that is used as a loss to minimize in the training loop.
    # several tensors can be returned for instance for analysis, see DeepARTrainingNetwork for an example.
    def create_training_network(
        self, device: torch.device
    ) -> SimpleFeedForwardTrainingNetwork:
        return SimpleFeedForwardTrainingNetwork(
            num_hidden_dimensions=self.num_hidden_dimensions,
            prediction_length=self.prediction_length,
            context_length=self.context_length,
            distr_output=self.distr_output,
            batch_normalization=self.batch_normalization,
            mean_scaling=self.mean_scaling,
        ).to(device)

    # we now define how the prediction happens given that we are provided a
    # training network.
    def create_predictor(
        self,
        transformation: Transformation,
        trained_network: nn.Module,
        device: torch.device,
    ) -> PTSPredictor:
        prediction_network = SimpleFeedForwardPredictionNetwork(
            num_hidden_dimensions=self.num_hidden_dimensions,
            prediction_length=self.prediction_length,
            context_length=self.context_length,
            distr_output=self.distr_output,
            batch_normalization=self.batch_normalization,
            mean_scaling=self.mean_scaling,
            num_parallel_samples=self.num_parallel_samples,
        ).to(device)

        copy_parameters(trained_network, prediction_network)

        return PTSPredictor(
            input_transform=transformation,
            prediction_net=prediction_network,
            batch_size=self.trainer.batch_size,
            freq=self.freq,
            prediction_length=self.prediction_length,
            device=device,
        )
