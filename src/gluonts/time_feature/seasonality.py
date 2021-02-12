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

logger = logging.getLogger(__name__)


DEFAULT_SEASONALITIES = {
    "H": 24,
    "D": 1,
    "W": 1,
    "M": 12,
    "B": 5,
    "Q": 4,
}


def get_seasonality(freq: str, seasonalities=DEFAULT_SEASONALITIES) -> int:
    """Return the seasonality of a given frequency:

    >>> get_seasonality("2H")
    12

    """
    offset = pd.tseries.frequencies.to_offset(freq)

    base_seasonality = seasonalities.get(norm_freq_str(offset.name), 1)

    seasonality, remainder = divmod(base_seasonality, offset.n)
    if not remainder:
        return seasonality

    logger.warning(
        f"Multiple {offset.n} does not divide base seasonality "
        f"{base_seasonality}. Falling back to seasonality 1."
    )
    return 1
