from typing import List, Optional

from pandas.tseries.frequencies import to_offset


def lags_for_fourier_time_features_from_frequency(
    freq_str: str, num_lags: Optional[int] = None
) -> List[int]:
    offset = to_offset(freq_str)
    multiple, granularity = offset.n, offset.name

    if granularity == "M":
        lags = [[1, 12]]
    elif granularity == "D":
        lags = [[1, 7, 14]]
    elif granularity == "B":
        lags = [[1, 2]]
    elif granularity == "H":
        lags = [[1, 24, 168]]
    elif granularity in ("T", "min"):
        lags = [[1, 4, 12, 24, 48]]
    else:
        lags = [[1]]

    # use less lags
    output_lags = list([int(lag) for sub_list in lags for lag in sub_list])
    output_lags = sorted(list(set(output_lags)))
    return output_lags[:num_lags]
