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

from gluonts.nursery.temporal_hierarchical_forecasting.utils.utils import (
    to_TemporalHierarchy,
)


TEMPORAL_HIERARCHIES = {
    "1min": to_TemporalHierarchy(freq_strs=["1H", "30min", "1min"]),
    "5min": to_TemporalHierarchy(freq_strs=["1H", "30min", "5min"]),
    "10min": to_TemporalHierarchy(freq_strs=["1H", "30min", "10min"]),
    "15min": to_TemporalHierarchy(freq_strs=["1H", "30min", "15min"]),
    "30min": to_TemporalHierarchy(freq_strs=["1H", "30min"]),
    "1H": to_TemporalHierarchy(freq_strs=["8H", "1H"]),
    "1D": to_TemporalHierarchy(freq_strs=["1W", "1D"]),
    "1B": to_TemporalHierarchy(freq_strs=["5D", "1D"]),
    "1W": to_TemporalHierarchy(freq_strs=["4W", "1W"]),
    "1M": to_TemporalHierarchy(freq_strs=["12M", "3M", "1M"]),
    "1Q": to_TemporalHierarchy(freq_strs=["12M", "3M"]),
}
TEMPORAL_HIERARCHIES["10T"] = TEMPORAL_HIERARCHIES["10min"]
TEMPORAL_HIERARCHIES["H"] = TEMPORAL_HIERARCHIES["1H"]
TEMPORAL_HIERARCHIES["D"] = TEMPORAL_HIERARCHIES["1D"]
TEMPORAL_HIERARCHIES["B"] = TEMPORAL_HIERARCHIES["1B"]
TEMPORAL_HIERARCHIES["M"] = TEMPORAL_HIERARCHIES["1M"]
TEMPORAL_HIERARCHIES["3M"] = TEMPORAL_HIERARCHIES["1Q"]
