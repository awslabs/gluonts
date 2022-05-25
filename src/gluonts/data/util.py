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

from typing import cast
import pandas as pd

from pandas.tseries.offsets import BaseOffset


def period_delta(a: pd.Period, b: pd.Period) -> int:
    """Calculate the delta between two `pandas.Period` objects.

    It should fulfill the invariant:

        a + period_delta(a, b) == b
    """
    assert a.freq == b.freq

    return cast(BaseOffset, a - b).n / a.freq.n
