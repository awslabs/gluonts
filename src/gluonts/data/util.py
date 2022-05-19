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
