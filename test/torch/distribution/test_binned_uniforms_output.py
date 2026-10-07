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


def test_binned_uniforms_output_can_be_constructed():
    distr_output = BinnedUniformsOutput(
        bins_lower_bound=0.0, bins_upper_bound=1.0, num_bins=4
    )
    assert distr_output.num_bins == 4
    assert distr_output.args_dim == {"logits": 4}


def test_binned_uniforms_output_distribution_from_args_proj():
    distr_output = BinnedUniformsOutput(
        bins_lower_bound=0.0, bins_upper_bound=1.0, num_bins=4
    )
    args_proj = distr_output.get_args_proj(in_features=8)
    distr_args = args_proj(torch.randn(2, 5, 8))

    distr = distr_output.distribution(distr_args)

    assert isinstance(distr, BinnedUniforms)
    assert distr.batch_shape == (2, 5)
    assert distr.sample().shape == (2, 5)
