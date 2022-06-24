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

from torch.distributions import Uniform

from gluonts.torch.distributions import AffineTransformed


def test_affine_transformed_uniform():
    a, b = 10.0, 15.0
    loc, scale = 20.0, 3.0
    distr = Uniform(a, b)
    transformed_distr = AffineTransformed(distr, loc=loc, scale=scale)

    assert transformed_distr.mean == (scale * (a + b) + 2 * loc) / 2
    assert transformed_distr.variance == (scale**2 * (b - a) ** 2) / 12
