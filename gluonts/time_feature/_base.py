# Third-party imports
import numpy as np
import pandas as pd

# First-party imports
from gluonts.core.component import validated


class TimeFeature:
    """
    A feature that only depends on time.
    """

    @validated()
    def __init__(self, normalized: bool = True):
        self.normalized = normalized

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        pass

    def __repr__(self):
        return self.__class__.__name__ + '()'


class MinuteOfHour(TimeFeature):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        if self.normalized:
            return index.minute / 59.0 - 0.5
        else:
            return index.minute.map(float)


class HourOfDay(TimeFeature):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        if self.normalized:
            return index.hour / 23.0 - 0.5
        else:
            return index.hour.map(float)


class DayOfWeek(TimeFeature):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        if self.normalized:
            return index.dayofweek / 6.0 - 0.5
        else:
            return index.dayofweek.map(float)


class DayOfMonth(TimeFeature):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        if self.normalized:
            return index.day / 30.0 - 0.5
        else:
            return index.day.map(float)


class DayOfYear(TimeFeature):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        if self.normalized:
            return index.dayofyear / 364.0 - 0.5
        else:
            return index.dayofyear.map(float)


class MonthOfYear(TimeFeature):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        if self.normalized:
            return index.month / 11.0 - 0.5
        else:
            return index.month.map(float)


class WeekOfYear(TimeFeature):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        if self.normalized:
            return index.weekofyear / 51.0 - 0.5
        else:
            return index.weekofyear.map(float)
