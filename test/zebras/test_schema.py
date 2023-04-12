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

import numpy as np

import gluonts.zebras as zb


schema = zb.Schema(
    {
        "target": zb.Field(ndim=1, tdim=0, past_only=True),
        "time_feat": zb.Field(ndim=2, tdim=-1, preprocess=np.atleast_2d),
        "static_feat": zb.Field(ndim=1, preprocess=np.atleast_1d),
    }
)


def test_schema_timeframe():
    xs = [1, 2, 3, 4, 5]
    row = {"target": xs, "time_feat": xs, "static_feat": 1}
    schema.load_timeframe(row)

    row = {"target": xs, "time_feat": [xs, xs], "static_feat": [1]}
    schema.load_timeframe(row)


def test_schema_timeframe_missing():
    xs = [1, 2, 3, 4, 5]
    row = {"target": xs, "time_feat": xs}

    with pytest.raises(Exception):
        schema.load_timeframe({})

    with pytest.raises(Exception):
        schema.load_timeframe(row)


def test_schema_splitframe():
    short = [1, 2, 3, 4, 5]
    long = [1, 2, 3, 4, 5, 6, 7, 8]
    row = {"target": short, "time_feat": long, "static_feat": 1}
    schema.load_splitframe(row)

    row = {"target": short, "time_feat": [long, long], "static_feat": [1]}
    schema.load_splitframe(row)


def test_schema_splitframe_error():
    xs = [1, 2, 3, 4, 5]

    with pytest.raises(Exception):
        schema.load_splitframe({})

    with pytest.raises(Exception):
        row = {"target": xs, "time_feat": xs}
        schema.load_timeframe(row)
