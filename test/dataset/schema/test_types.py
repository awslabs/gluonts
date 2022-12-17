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

import numpy as np
import pandas as pd

from gluonts.dataset.schema import types as ty

f32_2d = ty.Array(dtype=np.float32, ndim=2)


def test_array():
    assert np.array_equal(
        f32_2d([[0, 1, 2]]), np.arange(3, dtype=np.float32).reshape(1, 3)
    )


def test_period():
    assert pd.Period("2020", freq="M") == ty.Period(freq="M")("2020")
