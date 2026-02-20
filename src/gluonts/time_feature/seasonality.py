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


#: Default calendar-based seasonality values for common pandas frequencies.
#:
#: These are heuristic defaults based on common calendar conventions, not
#: values inferred from data. They are used when a seasonality parameter
#: is required but no data-driven estimation is performed.
#:
#: - Seconds ("S", "s"): 3600 (one hour of seconds)
#: - Minutes ("T", "min"): 1440 (one day of minutes)
#: - Hours ("H", "h"): 24 (one day of hours)
#: - Days ("D"): 1 (no sub-daily seasonality assumed)
#: - Weeks ("W"): 1 (no sub-weekly seasonality assumed)
#: - Months ("M", "ME"): 12 (one year of months)
#: - Business days ("B"): 5 (one week of business days)
#: - Quarters ("Q", "QE"): 4 (one year of quarters)
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


def get_seasonality(freq: str, seasonalities=DEFAULT_SEASONALITIES) -> int:
    """
    Return the default calendar-based seasonality for a given frequency.

    .. note::
        This function does **not** detect or estimate seasonality from data.
        It returns a predetermined constant based on the pandas frequency
        string, using common calendar heuristics (e.g., hourly data defaults
        to 24, monthly data defaults to 12).

    The returned value is used in evaluation metrics (e.g., MASE scaling),
    seasonal naive baselines, and other components that require a seasonality
    parameter but do not perform statistical inference.

    Parameters
    ----------
    freq
        A pandas-compatible frequency string (e.g., "H", "D", "M", "2H").
    seasonalities
        A dictionary mapping base frequency strings to their default
        seasonality values. Defaults to ``DEFAULT_SEASONALITIES``.

    Returns
    -------
    int
        The default seasonality for the given frequency. Returns 1 if the
        frequency is not recognized or the multiple does not evenly divide
        the base seasonality.

    Examples
    --------
    >>> get_seasonality("H")
    24
    >>> get_seasonality("2H")
    12
    >>> get_seasonality("M")
    12
    >>> get_seasonality("D")
    1

    See Also
    --------
    DEFAULT_SEASONALITIES : The mapping of base frequencies to seasonality.
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
