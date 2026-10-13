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

import json
import zipfile
from unittest import mock

from gluonts.dataset.jsonl import JsonLinesWriter
from gluonts.dataset.repository import _uber_tlc

# The raw file is not in time order, so the first row of a location is not
# its earliest pickup.
RAW_CSV = (
    "Dispatching_base_num,Pickup_date,Affiliated_base_num,locationID\n"
    "B1,2015-01-03 10:00:00,B1,7\n"
    "B1,2015-01-01 09:00:00,B1,7\n"
    "B1,2015-01-01 11:00:00,B1,7\n"
    "B1,2015-01-02 08:00:00,B1,7\n"
)


def _fake_urlretrieve(url, path):
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("uber-raw-data-janjune-15.csv", RAW_CSV)


def test_uber_tlc_start_matches_first_target_value(tmp_path):
    with mock.patch.object(
        _uber_tlc.request, "urlretrieve", _fake_urlretrieve
    ):
        _uber_tlc.generate_uber_dataset(
            tmp_path / "uber_tlc_daily",
            "daily",
            prediction_length=1,
            dataset_writer=JsonLinesWriter(use_gzip=False),
        )

    with open(tmp_path / "uber_tlc_daily" / "test" / "data.json") as f:
        (entry,) = [json.loads(line) for line in f]

    # daily pickup counts for 2015-01-01, 2015-01-02, 2015-01-03
    assert entry["target"] == [2, 1, 1]
    assert entry["start"].startswith("2015-01-01")
