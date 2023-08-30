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


from gluonts.dataset.common import ListDataset
from gluonts.dataset.schema import Translator


def test_translate():
    input_data = {"a": np.arange(100).reshape(20, 5)}

    t = Translator.parse(b="a.T[0]")
    b = t(input_data)["b"]

    assert np.array_equal(b, np.arange(0, 100, 5))


def test_dataset_translate():
    input_data = {
        "sales": np.random.random(100),
        "start": "2020",
        "price": np.full(100, 1.5),
        "temperature": np.full(100, 23),
    }

    for entry in ListDataset(
        [input_data],
        freq="D",
        translate={
            "target": "sales",
            "feat_dynamic_real": "[price, temperature]",
            "feat_dynamic_real_T": "[price, temperature].transpose()",
        },
    ):
        assert entry["target"].shape == (100,)
        assert entry["feat_dynamic_real"].shape == (2, 100)
        assert entry["feat_dynamic_real_T"].shape == (100, 2)
