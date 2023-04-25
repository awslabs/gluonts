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

import torch
from torch.distributions import (
    Distribution,
    TransformedDistribution,
    AffineTransform,
)


class ImplicitQuantile(Distribution):
    arg_constraints = {}

    def __init__(
        self,
        implicit_quantile_function,
        taus,
        nn_output,
        predicted_quantiles,
        validate_args=None,
    ):
        self.predicted_quantiles = predicted_quantiles[0]
        self.taus = taus
        self.quantile_function = implicit_quantile_function
        self.input_data = nn_output

        super(ImplicitQuantile, self).__init__(
            batch_shape=self.predicted_quantiles.shape,
            validate_args=validate_args,
        )

    @torch.no_grad()
    def sample(self, sample_shape=torch.Size()):
        """See arXiv: 1806.06923
        Once the model has learned how to predict a given quantile tau, one can sample from the
        distribution of the target, by sampling tau values.
        """
        if len(sample_shape) == 0:
            num_parallel_samples = 1
        else:
            num_parallel_samples = sample_shape[0]
        input_data = torch.repeat_interleave(
            self.input_data, repeats=num_parallel_samples, dim=0
        )
        batch_size = input_data.shape[0]
        forecast_length = input_data.shape[1]
        device = input_data.device

        taus = torch.rand(size=(batch_size, forecast_length), device=device)
        samples = self.quantile_function(input_data, taus)
        if len(sample_shape) == 0:
            return samples
        else:
            return samples.reshape((num_parallel_samples, -1, forecast_length))

    def log_prob(self, value):
        # Assumes same distribution for all steps in the future, conditionally on the input data
        return -self.quantile_loss(self.predicted_quantiles, value, self.taus)

    @staticmethod
    def quantile_loss(quantile_forecast, target, tau):
        return torch.abs(
            (quantile_forecast - target)
            * ((target <= quantile_forecast).float() - tau)
        )


class TransformedImplicitQuantile(TransformedDistribution):
    def __init__(self, base_distribution, transforms):
        super().__init__(base_distribution, transforms)

    def log_prob(self, x):
        scale = 1.0
        for transform in reversed(self.transforms):
            assert isinstance(
                transform, AffineTransform
            ), "Not an AffineTransform"
            x = transform.inv(x)
            scale *= transform.scale
        p = self.base_dist.log_prob(x)
        return p * scale
