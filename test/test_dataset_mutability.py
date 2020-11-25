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

# Third-party imports
import numpy as np
import pandas as pd
from copy import deepcopy

# First-party imports
from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
from gluonts.transform import (
    AddObservedValuesIndicator,
    Chain,
    AddAgeFeature,
    AddConstFeature,
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

ds1_c = deepcopy(ds1)
ds2_c = deepcopy(ds2)

transform = Chain(
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
    ]
)

# test that using twice the transformation gives the same result
for d in [ds1, ds2]:
    out1 = list(transform(d, is_train=True))
    out2 = list(transform(d, is_train=True))
    for o1, o2 in zip(out1, out2):
        for k in o1:
            if isinstance(o1[k], np.ndarray):
                assert np.all(np.isclose(o1[k], o2[k], equal_nan=True))
            else:
                assert o1[k] == o2[k]

# test that the original dataset did not change due to the transformation
for zip_per_type in [zip(ds1, ds1_c), zip(list(ds2), list(ds2_c))]:
    for d, d_c in zip_per_type:
        for k in d:
            if isinstance(d[k], np.ndarray):
                assert np.all(np.isclose(d[k], d_c[k], equal_nan=True))
            else:
                assert d[k] == d_c[k]
