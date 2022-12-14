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

from gluonts.util import pad_and_slice


def test_pad_and_slice_pad():
    data = np.arange(100).reshape(5, 20)

    d2, padded = pad_and_slice(
        data, 10, axis=0, pad_to="left", take_from="right"
    )
    assert d2.shape == (10, 20)
    assert padded.sum() == 20 - 10


def test_pad_and_slice_slice():
    data = np.arange(200).reshape(10, 20)

    d2, padded = pad_and_slice(
        data, 5, axis=0, pad_to="right", take_from="left"
    )
    assert d2.shape == (5, 20)
    assert padded.sum() == 0

    assert np.array_equal(d2, np.arange(100).reshape(5, 20))
