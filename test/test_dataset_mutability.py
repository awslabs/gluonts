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

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
from gluonts.transform import (
    AddAgeFeature,
    AddConstFeature,
    AddObservedValuesIndicator,
)

ds1 = [
    {
        "start": pd.Timestamp("2020/01/01", freq="1D"),
        "target": np.array(
            [1, 2, 3, np.nan, 5, np.nan, 7, np.nan, np.nan, 10]
        ),
    }
]
ds2 = ListDataset(
    [
        {
            "start": "2020/01/01",
            "target": [1, 2, 3, np.nan, 5, np.nan, 7, np.nan, np.nan, 10],
        }
    ],
    freq="1D",
)


@pytest.mark.parametrize("ds", [ds1, ds2])
@pytest.mark.parametrize(
    "transform",
    [
        AddObservedValuesIndicator(
            target_field=FieldName.TARGET,
            output_field=FieldName.OBSERVED_VALUES,
        ),
        AddAgeFeature(
            target_field=FieldName.TARGET,
            output_field="age_feature",
            pred_length=1,
        ),
        AddConstFeature(
            target_field=FieldName.TARGET,
            output_field="constant",
            pred_length=1,
        ),
    ],
)
def test_dataset_imutability(ds, transform):
    ds_c = deepcopy(ds)

    # test that using twice the transformation gives the same result
    out1 = transform(ds, is_train=True)
    out2 = transform(ds, is_train=True)
    for o1, o2 in zip(out1, out2):
        for k in o1:
            if isinstance(o1[k], np.ndarray):
                assert np.allclose(o1[k], o2[k], equal_nan=True)
            else:
                assert o1[k] == o2[k]

    # test that the original dataset did not change due to the transformation
    for d, d_c in zip(ds, ds_c):
        for k in d:
            if isinstance(d[k], np.ndarray):
                assert np.allclose(d[k], d_c[k], equal_nan=True)
            else:
                assert d[k] == d_c[k]
