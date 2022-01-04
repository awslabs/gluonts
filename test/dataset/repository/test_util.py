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

from gluonts.dataset.repository import _util


def test_metadata_1():
    assert _util.metadata(
        freq="1H", prediction_length=20, cardinality=[10, 3]
    ) == {
        "freq": "1H",
        "prediction_length": 20,
        "feat_static_cat": [
            {"name": "feat_static_cat_0", "cardinality": "10"},
            {"name": "feat_static_cat_1", "cardinality": "3"},
        ],
    }


def test_metadata_2():
    assert _util.metadata(freq="1H", prediction_length=2, cardinality=10) == {
        "freq": "1H",
        "prediction_length": 2,
        "feat_static_cat": [
            {"name": "feat_static_cat_0", "cardinality": "10"}
        ],
    }
