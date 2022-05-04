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

from typing import Dict, List, Optional, Tuple, cast

import torch
from torch.distributions import AffineTransform, TransformedDistribution
from torch.distributions.normal import Normal

from gluonts.core.component import validated

from .distribution_output import DistributionOutput


class MQF2Distribution(torch.distributions.Distribution):
    r"""
    Distribution class for the model MQF2 proposed in the paper
    ``Multivariate Quantile Function Forecaster``
    by Kan, Aubet, Januschowski, Park, Benidis, Ruthotto, Gasthaus

    Parameters
    ----------
    picnn
        A SequentialNet instance of a
        partially input convex neural network (picnn)
    hidden_state
        hidden_state obtained by unrolling the RNN encoder
        shape = (batch_size, context_length, hidden_size) in training
        shape = (batch_size, hidden_size) in inference
    prediction_length
        Length of the prediction horizon
    is_energy_score
        If True, use energy score as objective function
        otherwise use maximum likelihood as
        objective function (normalizing flows)
    es_num_samples
        Number of samples drawn to approximate the energy score
    beta
        Hyperparameter of the energy score (power of the two terms)
    threshold_input
        Clamping threshold of the (scaled) input when maximum
        likelihood is used as objective function
        this is used to make the forecaster more robust
        to outliers in training samples
    validate_args
        Sets whether validation is enabled or disabled
        For more details, refer to the descriptions in
        torch.distributions.distribution.Distribution
    """

    def __init__(
        self,
        picnn: torch.nn.Module,
        hidden_state: torch.Tensor,
        prediction_length: int,
        is_energy_score: bool = True,
        es_num_samples: int = 50,
        beta: float = 1.0,
        threshold_input: float = 100.0,
        validate_args: bool = False,
    ) -> None:

        self.picnn = picnn
        self.hidden_state = hidden_state
        self.prediction_length = prediction_length
        self.is_energy_score = is_energy_score
        self.es_num_samples = es_num_samples
        self.beta = beta
        self.threshold_input = threshold_input

        super().__init__(
            batch_shape=self.batch_shape, validate_args=validate_args
        )

        self.context_length = (
            self.hidden_state.shape[-2]
            if len(self.hidden_state.shape) > 2
            else 1
        )
        self.numel_batch = MQF2Distribution.get_numel(self.batch_shape)

        # mean zero and std one
        mu = torch.tensor(
            0, dtype=hidden_state.dtype, device=hidden_state.device
        )
        sigma = torch.ones_like(mu)
        self.standard_normal = Normal(mu, sigma)

    def stack_sliding_view(self, z: torch.Tensor) -> torch.Tensor:
        """
        Auxiliary function for loss computation.

        Unfolds the observations by sliding a window of size prediction_length
        over the observations z
        Then, reshapes the observations into a 2-dimensional tensor for
        further computation

        Parameters
        ----------
        z
            A batch of time series with shape
            (batch_size, context_length + prediction_length - 1)

        Returns
        -------
        Tensor
            Unfolded time series with shape
            (batch_size * context_length, prediction_length)
        """

        z = z.unfold(dimension=-1, size=self.prediction_length, step=1)
        z = z.reshape(-1, z.shape[-1])

        return z

    def loss(self, z: torch.Tensor) -> torch.Tensor:
        if self.is_energy_score:
            return self.energy_score(z)
        else:
            return -self.log_prob(z)

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        """
        Computes the log likelihood  log(g(z)) + logdet(dg(z)/dz), where g is
        the gradient of the picnn.

        Parameters
        ----------
        z
            A batch of time series with shape
            (batch_size, context_length + prediciton_length - 1)

        Returns
        -------
        loss
            Tesnor of shape (batch_size * context_length,)
        """

        z = torch.clamp(z, min=-self.threshold_input, max=self.threshold_input)
        z = self.stack_sliding_view(z)

        loss = self.picnn.logp(
            z, self.hidden_state.reshape(-1, self.hidden_state.shape[-1])
        )

        return loss

    def energy_score(self, z: torch.Tensor) -> torch.Tensor:
        """
        Computes the (approximated) energy score sum_i ES(g,z_i), where
        ES(g,z_i) =

        -1/(2*es_num_samples^2) * sum_{w,w'} ||w-w'||_2^beta
        + 1/es_num_samples * sum_{w''} ||w''-z_i||_2^beta,
        w's are samples drawn from the
        quantile function g(., h_i) (gradient of picnn),
        h_i is the hidden state associated with z_i,
        and es_num_samples is the number of samples drawn
        for each of w, w', w'' in energy score approximation

        Parameters
        ----------
        z
            A batch of time series with shape
            (batch_size, context_length + prediction_length - 1)

        Returns
        -------
        loss
            Tensor of shape (batch_size * context_length,)
        """

        es_num_samples = self.es_num_samples
        beta = self.beta

        z = self.stack_sliding_view(z)
        reshaped_hidden_state = self.hidden_state.reshape(
            -1, self.hidden_state.shape[-1]
        )

        loss = self.picnn.energy_score(
            z, reshaped_hidden_state, es_num_samples=es_num_samples, beta=beta
        )

        return loss

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        """
        Generates the sample paths.

        Parameters
        ----------
        sample_shape
            Shape of the samples

        Returns
        -------
        sample_paths
            Tesnor of shape (batch_size, *sample_shape, prediction_length)
        """

        numel_batch = self.numel_batch
        prediction_length = self.prediction_length

        num_samples_per_batch = MQF2Distribution.get_numel(sample_shape)
        num_samples = num_samples_per_batch * numel_batch

        hidden_state_repeat = self.hidden_state.repeat_interleave(
            repeats=num_samples_per_batch, dim=0
        )

        alpha = torch.rand(
            (num_samples, prediction_length),
            dtype=self.hidden_state.dtype,
            device=self.hidden_state.device,
            layout=self.hidden_state.layout,
        )

        return self.quantile(alpha, hidden_state_repeat).reshape(
            (numel_batch,) + sample_shape + (prediction_length,)
        )

    def quantile(
        self, alpha: torch.Tensor, hidden_state: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generates the predicted paths associated with the quantile levels
        alpha.

        Parameters
        ----------
        alpha
            quantile levels,
            shape = (batch_shape, prediction_length)
        hidden_state
            hidden_state, shape = (batch_shape, hidden_size)

        Returns
        -------
        results
            predicted paths of shape = (batch_shape, prediction_length)
        """

        if hidden_state is None:
            hidden_state = self.hidden_state

        normal_quantile = self.standard_normal.icdf(alpha)

        # In the energy score approach, we directly draw samples from picnn
        # In the MLE (Normalizing flows) approach, we need to invert the picnn
        # (go backward through the flow) to draw samples
        if self.is_energy_score:
            result = self.picnn(normal_quantile, context=hidden_state)
        else:
            result = self.picnn.reverse(normal_quantile, context=hidden_state)

        return result

    @staticmethod
    def get_numel(tensor_shape: torch.Size) -> int:
        # Auxiliary function
        # compute number of elements specified in a torch.Size()
        return torch.prod(torch.tensor(tensor_shape)).item()

    @property
    def batch_shape(self) -> torch.Size:
        # last dimension is the hidden state size
        return self.hidden_state.shape[:-1]

    @property
    def event_shape(self) -> Tuple:
        return ()

    @property
    def event_dim(self) -> int:
        return 0


class MQF2DistributionOutput(DistributionOutput):
    distr_cls: type = MQF2Distribution

    @validated()
    def __init__(
        self,
        prediction_length: int,
        is_energy_score: bool = True,
        threshold_input: float = 100.0,
        es_num_samples: int = 50,
        beta: float = 1.0,
    ) -> None:
        super().__init__(self)
        # A null args_dim to be called by PtArgProj
        self.args_dim = cast(
            Dict[str, int],
            {"null": 1},
        )

        self.prediction_length = prediction_length
        self.is_energy_score = is_energy_score
        self.threshold_input = threshold_input
        self.es_num_samples = es_num_samples
        self.beta = beta

    @classmethod
    def domain_map(
        cls,
        hidden_state: torch.Tensor,
    ) -> Tuple:
        # A null function to be called by ArgProj
        return ()

    def distribution(
        self,
        picnn: torch.nn.Module,
        hidden_state: torch.Tensor,
        loc: Optional[torch.Tensor] = 0,
        scale: Optional[torch.Tensor] = None,
    ) -> MQF2Distribution:

        distr = self.distr_cls(
            picnn,
            hidden_state,
            prediction_length=self.prediction_length,
            threshold_input=self.threshold_input,
            es_num_samples=self.es_num_samples,
            is_energy_score=self.is_energy_score,
            beta=self.beta,
        )

        if scale is None:
            return distr
        else:
            return TransformedMQF2Distribution(
                distr, [AffineTransform(loc=loc, scale=scale)]
            )

    @property
    def event_shape(self) -> Tuple:
        return ()


class TransformedMQF2Distribution(TransformedDistribution):
    @validated()
    def __init__(
        self,
        base_distribution: MQF2Distribution,
        transforms: List[AffineTransform],
        validate_args: bool = False,
    ) -> None:
        super().__init__(
            base_distribution, transforms, validate_args=validate_args
        )

    def scale_input(
        self, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Auxiliary function to scale the observations
        z = y
        scale = 1.0
        for t in self.transforms[::-1]:
            assert isinstance(t, AffineTransform), "Not an AffineTransform"
            z = t._inverse(y)
            scale *= t.scale

        return z, scale

    def repeat_scale(self, scale: torch.Tensor) -> torch.Tensor:
        return scale.squeeze(-1).repeat_interleave(
            self.base_dist.context_length, 0
        )

    def log_prob(self, y: torch.Tensor) -> torch.Tensor:
        prediction_length = self.base_dist.prediction_length

        z, scale = self.scale_input(y)
        p = self.base_dist.log_prob(z)

        repeated_scale = self.repeat_scale(scale)

        # the log scale term can be omitted
        # in optimization because it is a constant
        # prediction_length is the dimension of each sample
        return p - prediction_length * torch.log(repeated_scale)

    def energy_score(self, y: torch.Tensor) -> torch.Tensor:
        beta = self.base_dist.beta

        z, scale = self.scale_input(y)
        loss = self.base_dist.energy_score(z)

        repeated_scale = self.repeat_scale(scale)

        return loss * (repeated_scale**beta)
