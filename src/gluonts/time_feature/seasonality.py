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
import warnings

import pandas as pd

from gluonts.time_feature import norm_freq_str

logger = logging.getLogger(__name__)


DEFAULT_SEASONALITIES = {
    "S": 3600,  # 1 hour
    "s": 3600,  # 1 hour
    "T": 1440,  # 1 day
    "min": 1440,  # 1 day
    "H": 24,  # 1 day
    "h": 24,  # 1 day
    "D": 1,  # 1 day
    "W": 1,  # 1 week
    "M": 12,
    "ME": 12,
    "B": 5,
    "Q": 4,
    "QE": 4,
}


def get_seasonality_for_frequency(
    freq: str, seasonalities=DEFAULT_SEASONALITIES
) -> int:
    """
    Return a calendar-based default seasonality for the given frequency.

    This function does **not** inspect or analyse any time-series data.
    It maps a pandas frequency alias (e.g. ``"H"``, ``"D"``, ``"M"``) to a
    hard-coded calendar convention and divides by the interval multiplier.

    Examples
    --------
    >>> get_seasonality_for_frequency("2h")
    12

    Parameters
    ----------
    freq
        A pandas-compatible frequency string (e.g. ``"H"``, ``"30min"``,
        ``"D"``, ``"W"``, ``"M"``).
    seasonalities
        Optional override of the default seasonality mapping.

    Returns
    -------
    int
        The default seasonal period for the given frequency.  Falls back to
        ``1`` when the multiplier does not evenly divide the base seasonality.
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


def get_seasonality(freq: str, seasonalities=DEFAULT_SEASONALITIES) -> int:
    """
    Deprecated alias for :func:`get_seasonality_for_frequency`.

    .. deprecated::
        Use :func:`get_seasonality_for_frequency` instead.  This function
        will be removed in a future release.
    """
    warnings.warn(
        "get_seasonality is deprecated; use "
        "get_seasonality_for_frequency instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return get_seasonality_for_frequency(freq, seasonalities)
