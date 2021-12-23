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


import re
from functools import lru_cache
from typing import Tuple


def get_granularity(freq_str: str) -> Tuple[int, str]:
    """
    Splits a frequency string such as "7D" into the multiple 7 and the base
    granularity "D".

    Parameters
    ----------

    freq_str
        Frequency string of the form [multiple][granularity] such as "12H", "5min", "1D" etc.
    """
    freq_regex = r"\s*((\d+)?)\s*([^\d]\w*)"
    m = re.match(freq_regex, freq_str)
    assert m is not None, "Cannot parse frequency string: %s" % freq_str
    groups = m.groups()
    multiple = int(groups[1]) if groups[1] is not None else 1
    granularity = groups[2]
    return multiple, granularity


@lru_cache()
def get_seasonality(freq: str) -> int:
    """
    Returns the default seasonality for a given freq str. E.g. for

      2H -> 12

    """
    match = re.match(r"(\d*)(\w+)", freq)
    assert match, "Cannot match freq regex"
    mult, base_freq = match.groups()
    multiple = int(mult) if mult else 1

    seasonalities = {"H": 24, "D": 1, "W": 1, "M": 12, "B": 5}
    if base_freq in seasonalities:
        seasonality = seasonalities[base_freq]
    else:
        seasonality = 1
    if seasonality % multiple != 0:
        # logging.warning(
        #     f"multiple {multiple} does not divide base "
        #     f"seasonality {seasonality}."
        #     f"Falling back to seasonality 1"
        # )
        return 1
    return seasonality // multiple
