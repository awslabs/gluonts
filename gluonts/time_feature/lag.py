# Standard library imports
import re
from typing import List

# Third-party imports
import numpy as np

# First-party imports
from gluonts.time_feature import (
    DayOfMonth,
    DayOfWeek,
    DayOfYear,
    HourOfDay,
    MinuteOfHour,
    MonthOfYear,
    TimeFeature,
    WeekOfYear,
)


def get_granularity(freq_str):
    freq_regex = r'\s*((\d+)?)\s*([^\d]\w*)'
    m = re.match(freq_regex, freq_str)
    assert m is not None, "Cannot parse frequency string: %s" % freq_str
    groups = m.groups()
    multiple = int(groups[1]) if groups[1] is not None else 1
    granularity = groups[2]
    return multiple, granularity


def time_features_from_frequency_str(freq_str: str) -> List[TimeFeature]:
    multiple, granularity = get_granularity(freq_str)
    if granularity == 'M':
        feature_classes = [MonthOfYear]
    elif granularity == 'W':
        feature_classes = [DayOfMonth, WeekOfYear]
    elif granularity in ['D', 'B']:
        feature_classes = [DayOfWeek, DayOfMonth, DayOfYear]
    elif granularity == 'H':
        feature_classes = [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear]
    elif granularity in ['min', 'T']:
        feature_classes = [
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ]
    else:
        supported_freq_msg = f"""
        Unsupported frequency {freq_str}

        The following frequencies are supported:

            M   - monthly
            W   - week
            D   - daily
            H   - hourly
            min - minutely
        """
        raise RuntimeError(supported_freq_msg)

    return [cls() for cls in feature_classes]


def _make_lags(middle, delta):
    '''
    create a set of lags around a middle point including +/- delta
    '''
    return np.arange(middle - delta, middle + delta + 1).tolist()


def get_lags_for_frequency(freq_str, lag_ub=1200, num_lags=None):
    """
    By default all frequencies have the following lags: [1, 2, 3, 4, 5, 6, 7].
    Remaining lags correspond to the same `season` (+/- `delta`) in previous `k` cycles.
    Here `delta` and `k` are chosen according to the existing code.

    :param freq_str: [multiple]granularity
    :param lag_ub: the maximum value for a lag
    :param num_lags: upper bound on the number of lags; by default all generated lags are returned
    :return:
    """

    multiple, granularity = get_granularity(freq_str)

    # Lags are target values at the same `season` (+/- delta) but in the previous cycle.
    def _make_lags_for_minute(multiple, num_cycles=3):
        # We use previous ``num_cycles`` hours to generate lags
        return [
            _make_lags(k * 60 // multiple, 2) for k in range(1, num_cycles + 1)
        ]

    def _make_lags_for_hour(multiple, num_cycles=7):
        # We use previous ``num_cycles`` days to generate lags
        return [
            _make_lags(k * 24 // multiple, 1) for k in range(1, num_cycles + 1)
        ]

    def _make_lags_for_day(multiple, num_cycles=4):
        # We use previous ``num_cycles`` weeks to generate lags
        # We use the last month (in addition to 4 weeks) to generate lag.
        return [
            _make_lags(k * 7 // multiple, 1) for k in range(1, num_cycles + 1)
        ] + [_make_lags(30 // multiple, 1)]

    def _make_lags_for_week(multiple, num_cycles=3):
        # We use previous ``num_cycles`` years to generate lags
        # Additionally, we use previous 4, 8, 12 weeks
        return [
            _make_lags(k * 52 // multiple, 1) for k in range(1, num_cycles + 1)
        ] + [[4 // multiple, 8 // multiple, 12 // multiple]]

    def _make_lags_for_month(multiple, num_cycles=3):
        # We use previous ``num_cycles`` years to generate lags
        return [
            _make_lags(k * 12 // multiple, 1) for k in range(1, num_cycles + 1)
        ]

    if granularity == 'M':
        lags = _make_lags_for_month(multiple)
    elif granularity == 'W':
        lags = _make_lags_for_week(multiple)
    elif granularity == 'D':
        lags = _make_lags_for_day(multiple) + _make_lags_for_week(
            multiple / 7.0
        )
    elif granularity == 'B':
        # todo find good lags for business day
        lags = []
    elif granularity == 'H':
        lags = (
            _make_lags_for_hour(multiple)
            + _make_lags_for_day(multiple / 24.0)
            + _make_lags_for_week(multiple / (24.0 * 7))
        )
    elif granularity == 'min':
        lags = (
            _make_lags_for_minute(multiple)
            + _make_lags_for_hour(multiple / 60.0)
            + _make_lags_for_day(multiple / (60.0 * 24))
            + _make_lags_for_week(multiple / (60.0 * 24 * 7))
        )
    else:
        raise Exception('invalid frequency')

    # flatten lags list and filter
    lags = [
        int(lag) for sub_list in lags for lag in sub_list if 7 < lag <= lag_ub
    ]
    lags = [1, 2, 3, 4, 5, 6, 7] + sorted(list(set(lags)))

    return lags[:num_lags]
