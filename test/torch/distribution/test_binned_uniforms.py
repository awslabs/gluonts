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

from gluonts.torch.distributions import BinnedUniforms


@pytest.mark.parametrize(
    "logits, expected_mean",
    [
        # equal bin probabilities: uniform on [-2, 2]
        (torch.zeros(3, 4), torch.zeros(3)),
        # all mass in the last bin [1, 2]
        (torch.tensor([[-100.0, -100.0, -100.0, 0.0]]), torch.tensor([1.5])),
        # bins [-2, -1] and [1, 2] with probabilities 1/4 and 3/4
        (
            torch.tensor(
                [[0.0, -100.0, -100.0, torch.log(torch.tensor(3.0))]]
            ),
            torch.tensor([0.75]),
        ),
    ],
)
def test_binned_uniforms_mean(logits, expected_mean):
    distr = BinnedUniforms(
        bins_lower_bound=-2.0,
        bins_upper_bound=2.0,
        logits=logits,
        numb_bins=4,
    )
    torch.testing.assert_close(distr.mean, expected_mean)
