# Standard library imports
from typing import List

# Third-party imports
import mxnet as mx

# First-party imports
from gluonts.block.scaler import MeanScaler, NOPScaler
from gluonts.core.component import validated
from gluonts.distribution import Distribution, DistributionOutput
from gluonts.model.common import Tensor


class SimpleFeedForwardNetworkBase(mx.gluon.HybridBlock):
    """
    Defines a Gluon block used for training and predictions.
    """

    # This class does not implement hybrid_forward: this is delegated
    # to the two subclasses SimpleFeedForwardTrainingNetwork and
    # SimpleFeedForwardPredictionNetwork, that define respectively how to
    # compute the loss and how to generate predictions

    # Needs the validated decorator so that arguments types are checked and
    # the block can be serialized.
    @validated()
    def __init__(
        self,
        num_hidden_dimensions: List[int],
        prediction_length: int,
        context_length: int,
        batch_normalization: bool,
        mean_scaling: bool,
        distr_output: DistributionOutput,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.num_hidden_dimensions = num_hidden_dimensions
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.batch_normalization = batch_normalization
        self.mean_scaling = mean_scaling
        self.distr_output = distr_output

        with self.name_scope():
            self.distr_args_proj = self.distr_output.get_args_proj()
            self.mlp = mx.gluon.nn.HybridSequential()
            dims = self.num_hidden_dimensions
            for layer_no, units in enumerate(dims[:-1]):
                self.mlp.add(mx.gluon.nn.Dense(units=units, activation='relu'))
                if self.batch_normalization:
                    self.mlp.add(mx.gluon.nn.BatchNorm())
            self.mlp.add(mx.gluon.nn.Dense(units=prediction_length * dims[-1]))
            self.mlp.add(
                mx.gluon.nn.HybridLambda(
                    lambda F, o: F.reshape(
                        o, (-1, prediction_length, dims[-1])
                    )
                )
            )
            self.scaler = MeanScaler() if mean_scaling else NOPScaler()

    def get_distr(self, F, past_target: Tensor) -> Distribution:

        # (batch_size, seq_len, target_dim) and (batch_size, seq_len, target_dim)
        scaled_target, target_scale = self.scaler(
            past_target,
            F.ones_like(past_target),  # TODO: pass the actual observed here
        )
        mlp_outputs = self.mlp(scaled_target)
        distr_args = self.distr_args_proj(mlp_outputs)
        return self.distr_output.distribution(
            distr_args, scale=target_scale.expand_dims(axis=1)
        )


class SimpleFeedForwardTrainingNetwork(SimpleFeedForwardNetworkBase):
    # noinspection PyMethodOverriding,PyPep8Naming
    def hybrid_forward(
        self, F, past_target: Tensor, future_target: Tensor
    ) -> Tensor:
        """

        Parameters
        ----------
        F
        past_target : (batch_size, context_length, target_dim)
        future_target : (batch_size, prediction_length, target_dim)

        Returns
        -------

        """
        distr = self.get_distr(F, past_target)

        # (batch_size, prediction_length, target_dim)
        loss = distr.loss(future_target)

        # (batch_size, target_dim)
        return loss.mean(axis=1)


class SimpleFeedForwardPredictionNetwork(SimpleFeedForwardNetworkBase):
    @validated()
    def __init__(self, num_sample_paths: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_sample_paths = num_sample_paths

    # noinspection PyMethodOverriding,PyPep8Naming
    def hybrid_forward(self, F, past_target: Tensor) -> Tensor:
        """

        Parameters
        ----------
        F
        past_target : (batch_size, context_length, target_dim)

        Returns samples with shape (samples, batch_size, prediction_length,)
        -------

        """
        distr = self.get_distr(F, past_target)

        # (num_samples, batch_size, prediction_length)
        samples = distr.sample(self.num_sample_paths)

        # (batch_size, num_samples, prediction_length)
        return samples.swapaxes(0, 1)
