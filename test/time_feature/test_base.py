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

from pandas.tseries.frequencies import to_offset

from gluonts.time_feature import norm_freq_str


def test_norm_freq_str():
    assert norm_freq_str(to_offset("Y").name) in ["A", "YE"]
    assert norm_freq_str(to_offset("YS").name) in ["A", "Y"]
    assert norm_freq_str(to_offset("A").name) in ["A", "YE"]
    assert norm_freq_str(to_offset("AS").name) in ["A", "Y"]

    assert norm_freq_str(to_offset("Q").name) in ["Q", "QE"]
    assert norm_freq_str(to_offset("QS").name) == "Q"

    assert norm_freq_str(to_offset("M").name) in ["M", "ME"]
    assert norm_freq_str(to_offset("MS").name) in ["M", "ME"]

    assert norm_freq_str(to_offset("S").name) in ["S", "s"]
