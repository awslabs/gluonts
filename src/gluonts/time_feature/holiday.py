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

from typing import List, Callable

import numpy as np
import pandas as pd
from pandas.tseries.holiday import (
    SU,
    EasterMonday,
    GoodFriday,
    Holiday,
    USColumbusDay,
    USLaborDay,
    USMartinLutherKingJr,
    USMemorialDay,
    USPresidentsDay,
    USThanksgivingDay,
)
from pandas.tseries.offsets import DateOffset, Day, Easter

MAX_WINDOW = (
    183
)  # This is 183 to cover half a year (in both directions), also for leap years


def distance_to_holiday(holiday):
    def distance_to_day(index):
        holiday_date = holiday.dates(
            index - pd.Timedelta(days=MAX_WINDOW),
            index + pd.Timedelta(days=MAX_WINDOW),
        )
        assert (
            len(holiday_date) != 0
        ), f"No closest holiday for the date index {index} found."
        # It sometimes returns two dates if it is exactly half a year after the
        # holiday. In this case, the smaller distance (182 days) is returned.
        return (index - holiday_date[0]).days

    return distance_to_day


EasterSunday = Holiday(
    "Easter Sunday", month=1, day=1, offset=[Easter(), Day(0)]
)
NewYearsDay = Holiday('New Years Day', month=1, day=1)
SuperBowl = Holiday(
    "Superbowl", month=2, day=1, offset=DateOffset(weekday=SU(1))
)
MothersDay = Holiday(
    "Mothers Day", month=5, day=1, offset=DateOffset(weekday=SU(2))
)
IndependenceDay = Holiday('Independence Day', month=7, day=4)
ChristmasEve = Holiday('Christmas', month=12, day=24)
ChristmasDay = Holiday('Christmas', month=12, day=25)
NewYearsEve = Holiday('New Years Eve', month=12, day=31)


NEW_YEARS_DAY = 'new_years_day'
MARTIN_LUTHER_KING_DAY = 'martin_luther_king_day'
SUPERBOWL = 'superbowl'
PRESIDENTS_DAY = 'presidents_day'
GOOD_FRIDAY = 'good_friday'
EASTER_SUNDAY = 'easter_sunday'
EASTER_MONDAY = 'easter_monday'
MOTHERS_DAY = 'mothers_day'
INDEPENDENCE_DAY = 'independence_day'
LABOR_DAY = 'labor_day'
MEMORIAL_DAY = 'memorial_day'
COLUMBUS_DAY = 'columbus_day'
THANKSGIVING = 'thanksgiving'
CHRISTMAS_EVE = 'christmas_eve'
CHRISTMAS_DAY = 'christmas_day'
NEW_YEARS_EVE = 'new_years_eve'


SPECIAL_DATE_FEATURES = {
    NEW_YEARS_DAY: distance_to_holiday(NewYearsDay),
    MARTIN_LUTHER_KING_DAY: distance_to_holiday(USMartinLutherKingJr),
    SUPERBOWL: distance_to_holiday(SuperBowl),
    PRESIDENTS_DAY: distance_to_holiday(USPresidentsDay),
    GOOD_FRIDAY: distance_to_holiday(GoodFriday),
    EASTER_SUNDAY: distance_to_holiday(EasterSunday),
    EASTER_MONDAY: distance_to_holiday(EasterMonday),
    MOTHERS_DAY: distance_to_holiday(MothersDay),
    INDEPENDENCE_DAY: distance_to_holiday(IndependenceDay),
    LABOR_DAY: distance_to_holiday(USLaborDay),
    MEMORIAL_DAY: distance_to_holiday(USMemorialDay),
    COLUMBUS_DAY: distance_to_holiday(USColumbusDay),
    THANKSGIVING: distance_to_holiday(USThanksgivingDay),
    CHRISTMAS_EVE: distance_to_holiday(ChristmasEve),
    CHRISTMAS_DAY: distance_to_holiday(ChristmasDay),
    NEW_YEARS_EVE: distance_to_holiday(NewYearsEve),
}


# Kernel functions
def indicator(distance):
    return float(distance == 0)


def exponential_kernel(alpha=1.0, tol=1e-9):
    def kernel(distance):
        kernel_value = np.exp(-alpha * np.abs(distance))
        if kernel_value > tol:
            return kernel_value
        else:
            return 0.0

    return kernel


def squared_exponential_kernel(alpha=1.0, tol=1e-9):
    def kernel(distance):
        kernel_value = np.exp(-alpha * np.abs(distance) ** 2)
        if kernel_value > tol:
            return kernel_value
        else:
            return 0.0

    return kernel


class SpecialDateFeatureSet:
    """
    Implements calculation of holiday features. The SpecialDateFeatureSet is
    applied on a pandas Series with Datetimeindex and returns a 2D array of
    the shape (len(dates), num_features), where num_features are the number
    of holidays.

    Note that for lower than daily granularity the distance to the holiday is
    still computed on a per-day basis.

    Example use:

        >>> from gluonts.time_feature.holiday import (
        ...    squared_exponential_kernel,
        ...    SpecialDateFeatureSet,
        ...    CHRISTMAS_DAY,
        ...    CHRISTMAS_EVE
        ... )
        >>> import pandas as pd
        >>> sfs = SpecialDateFeatureSet([CHRISTMAS_EVE, CHRISTMAS_DAY])
        >>> date_indices = pd.date_range(
        ...     start="2016-12-24",
        ...     end="2016-12-31",
        ...     freq='D'
        ... )
        >>> sfs(date_indices)
        array([[1., 0., 0., 0., 0., 0., 0., 0.],
               [0., 1., 0., 0., 0., 0., 0., 0.]])

    Example use for using a squared exponential kernel:

        >>> kernel = squared_exponential_kernel(alpha=1.0)
        >>> sfs = SpecialDateFeatureSet([CHRISTMAS_EVE, CHRISTMAS_DAY], kernel)
        >>> sfs(date_indices)
        array([[1.00000000e+00, 3.67879441e-01, 1.83156389e-02, 1.23409804e-04,
                1.12535175e-07, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
               [3.67879441e-01, 1.00000000e+00, 3.67879441e-01, 1.83156389e-02,
                1.23409804e-04, 1.12535175e-07, 0.00000000e+00, 0.00000000e+00]])

    """

    def __init__(
        self,
        feature_names: List[str],
        kernel_function: Callable[[int], int] = indicator,
    ):
        """
        Parameters
        ----------
        feature_names
            list of strings with holiday names for which features should be created.
        kernel_function
            kernel function to pass the feature value based
            on distance in days. Can be indicator function (default),
            exponential_kernel, squared_exponential_kernel or user defined.
        """
        self.feature_names = feature_names
        self.num_features = len(feature_names)
        self.kernel_function = kernel_function

    def __call__(self, dates):
        """
        Transform a pandas series with timestamps to holiday features.

        Parameters
        ----------
        dates
            Pandas series with Datetimeindex timestamps.
        """
        return np.vstack(
            [
                np.hstack(
                    [
                        self.kernel_function(
                            SPECIAL_DATE_FEATURES[feat_name](index)
                        )
                        for index in dates
                    ]
                )
                for feat_name in self.feature_names
            ]
        )
