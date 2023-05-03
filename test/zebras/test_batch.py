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


def test_batch_time_series():
    ts0 = zb.time_series(
        [1, 2, 3], start="2020", freq="D", metadata={"item_id": 0}, name="x"
    )
    ts1 = zb.time_series([4, 5, 6])

    batched = zb.batch([ts0, ts1])

    assert len(batched) == 3
    assert batched.batch_size == 2

    ts0x, ts1x = batched.items()

    np.testing.assert_array_equal(ts0, ts0x)
    assert ts0.metadata == ts0x.metadata
    assert ts0.tdim == ts0x.tdim
    assert ts0.name == ts0x.name
    assert ts0._pad == ts0x._pad

    np.testing.assert_array_equal(ts1, ts1x)
    assert ts1.metadata == ts1x.metadata
    assert ts1.tdim == ts1x.tdim
    assert ts1.name == ts1x.name
    assert ts1._pad == ts1x._pad


def test_batch_time_frame():
    tf0 = zb.time_frame(
        {"x": [1, 2, 3]},
        start="2020",
        freq="D",
        metadata={"item_id": 0},
    )
    tf1 = zb.time_frame({"x": [4, 5, 6]})

    batched = zb.batch([tf0, tf1])

    assert len(batched) == 3
    assert batched.batch_size == 2

    tf0x, tf1x = batched.items()


def test_batch_split_frame():
    sf0 = zb.time_frame(
        {"x": [1, 2, 3]},
        start="2020",
        freq="D",
        metadata={"item_id": 0},
    ).split(1, past_length=4, future_length=3)
    sf1 = zb.time_frame({"x": [4, 5, 6]}).split(
        1, past_length=4, future_length=3
    )

    batched = zb.batch([sf0, sf1])

    assert len(batched) == 7
    assert batched.batch_size == 2

    sf0x, sf1x = batched.items()
