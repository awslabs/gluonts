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

import pytest
import torch

from gluonts.torch.distributions import BinnedUniforms, SplicedBinnedPareto


@pytest.mark.parametrize("sample_shape", [torch.Size(), torch.Size((3,))])
def test_binned_uniforms_sample_follows_parameter_dtype(sample_shape):
    distr = BinnedUniforms(
        bins_lower_bound=0.0,
        bins_upper_bound=1.0,
        logits=torch.zeros(2, 4, dtype=torch.float64),
        numb_bins=4,
    )
    samples = distr.sample(sample_shape)

    assert samples.shape == sample_shape + distr.batch_shape
    assert samples.dtype == torch.float64
    assert distr.median.dtype == torch.float64


def test_spliced_binned_pareto_sample_follows_parameter_dtype():
    args = dict(
        logits=torch.zeros(2, 4, dtype=torch.float64),
        upper_gp_xi=torch.full((2, 1), 0.1, dtype=torch.float64),
        upper_gp_beta=torch.ones(2, 1, dtype=torch.float64),
        lower_gp_xi=torch.full((2, 1), 0.1, dtype=torch.float64),
        lower_gp_beta=torch.ones(2, 1, dtype=torch.float64),
    )
    distr = SplicedBinnedPareto(
        bins_lower_bound=0.0, bins_upper_bound=1.0, numb_bins=4, **args
    )

    assert distr.sample((3,)).dtype == torch.float64
