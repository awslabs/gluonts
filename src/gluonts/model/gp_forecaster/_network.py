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

from typing import Tuple

import mxnet as mx

from gluonts.core.component import DType, validated
from gluonts.mx import Tensor
from gluonts.mx.distribution.distribution import softplus
from gluonts.mx.kernels import KernelOutputDict

from .gaussian_process import GaussianProcess


class GaussianProcessNetworkBase(mx.gluon.HybridBlock):
    """
    Defines a Gluon block used for GP training and predictions.
    """

    # The two subclasses GaussianProcessTrainingNetwork and
    # GaussianProcessPredictionNetwork define how to
    # compute the loss and how to generate predictions, respectively.

    @validated()
    def __init__(
        self,
        prediction_length: int,
        context_length: int,
        cardinality: int,
        kernel_output: KernelOutputDict,
        params_scaling: bool,
        float_type: DType,
        max_iter_jitter: int,
        jitter_method: str,
        **kwargs,
    ) -> None:
        """
        Parameters
        ----------
        prediction_length
            Prediction length.
        context_length
            Training length.
        cardinality
            Number of time series.
        kernel_output
            KernelOutput instance to determine which kernel subclass to be instantiated.
        params_scaling
            Determines whether or not to scale the model parameters.
        float_type
            Determines whether to use single or double precision.
        max_iter_jitter
            Maximum number of iterations for jitter to iteratively make the matrix positive definite.
        jitter_method
            Iteratively jitter method or use eigenvalue decomposition depending on problem size.
        **kwargs
            Arbitrary keyword arguments.
        """
        super().__init__(**kwargs)

        self.prediction_length = prediction_length
        self.context_length = context_length
        self.cardinality = cardinality
        self.kernel_output = kernel_output
        self.params_scaling = params_scaling
        self.float_type = float_type
        self.max_iter_jitter = max_iter_jitter
        self.jitter_method = jitter_method

        with self.name_scope():
            self.proj_kernel_args = kernel_output.get_args_proj(
                self.float_type
            )
            self.num_hyperparams = kernel_output.get_num_args()
            self.embedding = mx.gluon.nn.Embedding(
                # Noise sigma is additional parameter so add 1 to output dim
                input_dim=self.cardinality,
                output_dim=self.num_hyperparams + 1,
                dtype=self.float_type,
            )

    # noinspection PyMethodOverriding,PyPep8Naming
    def get_gp_params(
        self,
        F,
        past_target: Tensor,
        past_time_feat: Tensor,
        feat_static_cat: Tensor,
    ) -> Tuple:
        """
        This function returns the GP hyper-parameters for the model.

        Parameters
        ----------
        F
            A module that can either refer to the Symbol API or the NDArray
            API in MXNet.
        past_target
            Training time series values of shape (batch_size, context_length).
        past_time_feat
            Training features of shape (batch_size, context_length, num_features).
        feat_static_cat
            Time series indices of shape (batch_size, 1).

        Returns
        -------
        Tuple
            Tuple of kernel hyper-parameters of length num_hyperparams.
                Each is a Tensor of shape (batch_size, 1, 1).
            Model noise sigma.
                Tensor of shape (batch_size, 1, 1).
        """
        output = self.embedding(
            feat_static_cat.squeeze()
        )  # Shape (batch_size, num_hyperparams + 1)
        kernel_args = self.proj_kernel_args(output)
        sigma = softplus(
            F,
            output.slice_axis(  # sigma is the last hyper-parameter
                axis=1,
                begin=self.num_hyperparams,
                end=self.num_hyperparams + 1,
            ),
        )
        if self.params_scaling:
            scalings = self.kernel_output.gp_params_scaling(
                F, past_target, past_time_feat
            )
            sigma = F.broadcast_mul(sigma, scalings[self.num_hyperparams])
            kernel_args = (
                F.broadcast_mul(kernel_arg, scaling)
                for kernel_arg, scaling in zip(
                    kernel_args, scalings[0 : self.num_hyperparams]
                )
            )
        min_value = 1e-5
        max_value = 1e8
        kernel_args = (
            kernel_arg.clip(min_value, max_value).expand_dims(axis=2)
            for kernel_arg in kernel_args
        )
        sigma = sigma.clip(min_value, max_value).expand_dims(axis=2)
        return kernel_args, sigma


class GaussianProcessTrainingNetwork(GaussianProcessNetworkBase):
    # noinspection PyMethodOverriding,PyPep8Naming
    @validated()
    def __init__(self, *args, **kwargs) -> None:
        """
        Parameters
        ----------
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)

    # noinspection PyMethodOverriding,PyPep8Naming
    def hybrid_forward(
        self,
        F,
        past_target: Tensor,
        past_time_feat: Tensor,
        feat_static_cat: Tensor,
    ) -> Tensor:
        """
        Parameters
        ----------
        F
            A module that can either refer to the Symbol API or the NDArray
            API in MXNet.
        past_target
            Training time series values of shape (batch_size, context_length).
        past_time_feat
            Training features of shape (batch_size, context_length, num_features).
        feat_static_cat
            Time series indices of shape (batch_size, 1).
        Returns
        -------
        Tensor
            GP loss of shape (batch_size, 1)
        """
        kernel_args, sigma = self.get_gp_params(
            F, past_target, past_time_feat, feat_static_cat
        )
        kernel = self.kernel_output.kernel(kernel_args)
        gp = GaussianProcess(
            sigma=sigma,
            kernel=kernel,
            context_length=self.context_length,
            float_type=self.float_type,
            max_iter_jitter=self.max_iter_jitter,
            jitter_method=self.jitter_method,
        )
        return gp.log_prob(past_time_feat, past_target)


class GaussianProcessPredictionNetwork(GaussianProcessNetworkBase):
    @validated()
    def __init__(
        self, num_parallel_samples: int, sample_noise: bool, *args, **kwargs
    ) -> None:
        r"""
        Parameters
        ----------
        num_parallel_samples
            Number of samples to be drawn.
        sample_noise
            Boolean to determine whether to add :math:`\sigma^2I` to the predictive covariance matrix.
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.num_parallel_samples = num_parallel_samples
        self.sample_noise = sample_noise

    # noinspection PyMethodOverriding,PyPep8Naming
    def hybrid_forward(
        self,
        F,
        past_target: Tensor,
        past_time_feat: Tensor,
        future_time_feat: Tensor,
        feat_static_cat: Tensor,
    ) -> Tensor:
        """
        Parameters
        ----------
        F
            A module that can either refer to the Symbol API or the NDArray
            API in MXNet.
        past_target
            Training time series values of shape (batch_size, context_length).
        past_time_feat
            Training features of shape (batch_size, context_length, num_features).
        future_time_feat
            Test features of shape (batch_size, prediction_length, num_features).
        feat_static_cat
            Time series indices of shape (batch_size, 1).
        Returns
        -------
        Tensor
            GP samples of shape (batch_size, num_samples, prediction_length).
        """
        kernel_args, sigma = self.get_gp_params(
            F, past_target, past_time_feat, feat_static_cat
        )
        gp = GaussianProcess(
            sigma=sigma,
            kernel=self.kernel_output.kernel(kernel_args),
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            num_samples=self.num_parallel_samples,
            float_type=self.float_type,
            max_iter_jitter=self.max_iter_jitter,
            jitter_method=self.jitter_method,
            sample_noise=self.sample_noise,
        )
        samples, _, _ = gp.exact_inference(
            past_time_feat, past_target, future_time_feat
        )  # Shape (batch_size, prediction_length, num_samples)
        return samples.swapaxes(1, 2)
