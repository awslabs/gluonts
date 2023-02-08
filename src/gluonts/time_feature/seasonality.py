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

import logging

import pandas as pd

from gluonts.time_feature import norm_freq_str
from gluonts import zebras as zb

logger = logging.getLogger(__name__)


DEFAULT_SEASONALITIES = {
    "S": 3600,  # 1 hour
    "T": 1440,  # 1 day
    "H": 24,  # 1 day
    "D": 1,  # 1 day
    "W": 1,  # 1 week
    "M": 12,
    "B": 5,
    "Q": 4,
}


def get_seasonality(freq: str, seasonalities=DEFAULT_SEASONALITIES) -> int:
    """
    Return the seasonality of a given frequency:

    >>> get_seasonality("2H")
    12
    """
    freq = zb.freq(freq)
    base_seasonality = seasonalities.get(freq.pd_freq, 1)

    seasonality, remainder = divmod(base_seasonality, freq.multiple)
    if not remainder:
        return seasonality

    logger.warning(
        f"Multiple {offset.n} does not divide base seasonality "
        f"{base_seasonality}. Falling back to seasonality 1."
    )
    return 1
