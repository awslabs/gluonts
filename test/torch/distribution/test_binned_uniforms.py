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

from gluonts.torch.distributions import BinnedUniforms, BinnedUniformsOutput


def test_binned_uniforms_output_constructs():
    distr_output = BinnedUniformsOutput(
        bins_lower_bound=0.0, bins_upper_bound=1.0, num_bins=2
    )
    assert distr_output.num_bins == 2
    assert distr_output.beta == 0.0


def test_binned_uniforms_output_distribution_from_args_proj():
    """DeepAR (and other estimators) pass `distr_args` as a tuple of tensors
    from `get_args_proj`. `BinnedUniforms` needs the unpacked logits tensor.
    """
    num_bins = 2
    batch_size, time_length, in_features = 4, 8, 5
    prediction_length = 3

    distr_output = BinnedUniformsOutput(
        bins_lower_bound=0.0, bins_upper_bound=1.0, num_bins=num_bins
    )
    args_proj = distr_output.get_args_proj(in_features=in_features)
    net_out = torch.randn(batch_size, time_length, in_features)
    distr_args = args_proj(net_out)

    assert isinstance(distr_args, tuple)
    assert len(distr_args) == 1
    (logits,) = distr_args
    assert logits.shape == (batch_size, time_length, num_bins)

    # DeepAR slices the last `prediction_length` steps into a list/tuple.
    sliced_params = [p[:, -prediction_length:] for p in distr_args]
    distr = distr_output.distribution(sliced_params)

    assert isinstance(distr, BinnedUniforms)
    assert distr.batch_shape == (batch_size, prediction_length)

    target = torch.rand(batch_size, prediction_length)
    loss = distr_output.loss(target=target, distr_args=tuple(sliced_params))
    assert loss.shape == (batch_size, prediction_length)
    assert torch.isfinite(loss).all()
