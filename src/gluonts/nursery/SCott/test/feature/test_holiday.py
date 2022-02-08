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

# Third-party imports
import numpy as np
import pandas as pd
import pytest
from pandas.tseries.holiday import Holiday


# First-party imports
from pts.feature.holiday import (
    CHRISTMAS_DAY,
    CHRISTMAS_EVE,
    COLUMBUS_DAY,
    EASTER_MONDAY,
    EASTER_SUNDAY,
    GOOD_FRIDAY,
    INDEPENDENCE_DAY,
    LABOR_DAY,
    MARTIN_LUTHER_KING_DAY,
    MEMORIAL_DAY,
    MOTHERS_DAY,
    NEW_YEARS_DAY,
    NEW_YEARS_EVE,
    PRESIDENTS_DAY,
    SPECIAL_DATE_FEATURES,
    SUPERBOWL,
    THANKSGIVING,
    BLACK_FRIDAY,
    CYBER_MONDAY,
    SpecialDateFeatureSet,
    squared_exponential_kernel,
    exponential_kernel,
    CustomDateFeatureSet,
    CustomHolidayFeatureSet,
)

test_dates = {
    NEW_YEARS_DAY: [
        "2015-01-01",
        "2016-01-01",
        "2017-01-01",
        "2018-01-01",
        "2019-01-01",
    ],
    MARTIN_LUTHER_KING_DAY: [
        "2012-01-16",
        "2014-01-20",
        "2015-01-19",
        "2018-01-15",
        "2019-01-21",
    ],
    SUPERBOWL: ["2011-02-06", "2017-02-05", "2018-02-04", "2019-02-03"],
    PRESIDENTS_DAY: ["2011-02-21", "2017-02-20", "2018-02-19", "2019-02-18"],
    MEMORIAL_DAY: [
        "2015-05-25",
        "2016-05-30",
        "2017-05-29",
        "2018-05-28",
        "2019-05-27",
    ],
    GOOD_FRIDAY: [
        "2014-04-18",
        "2015-04-03",
        "2017-04-14",
        "2018-03-30",
        "2019-04-19",
    ],
    EASTER_SUNDAY: [
        "2014-04-20",
        "2015-04-05",
        "2017-04-16",
        "2018-04-01",
        "2019-04-21",
    ],
    EASTER_MONDAY: [
        "2014-04-21",
        "2015-04-06",
        "2017-04-17",
        "2018-04-02",
        "2019-04-22",
    ],
    MOTHERS_DAY: ["2016-05-08", "2017-05-14", "2018-05-13", "2019-05-12"],
    INDEPENDENCE_DAY: ["2016-07-04", "2017-07-04", "2018-07-04", "2019-07-04"],
    LABOR_DAY: ["2014-09-01", "2016-09-05", "2018-09-03", "2019-09-02"],
    COLUMBUS_DAY: ["2016-10-10", "2017-10-09", "2018-10-08", "2019-10-14"],
    THANKSGIVING: [
        "2015-11-26",
        "2016-11-24",
        "2017-11-23",
        "2018-11-22",
        "2019-11-28",
    ],
    CHRISTMAS_EVE: ["2016-12-24", "2017-12-24", "2018-12-24", "2019-12-24"],
    CHRISTMAS_DAY: ["2016-12-25", "2017-12-25", "2018-12-25", "2019-12-25"],
    NEW_YEARS_EVE: ["2016-12-31", "2017-12-31", "2018-12-31", "2019-12-31"],
    BLACK_FRIDAY: [
        "2016-11-25",
        "2017-11-24",
        "2018-11-23",
        "2019-11-29",
        "2020-11-27",
    ],
    CYBER_MONDAY: [
        "2016-11-28",
        "2017-11-27",
        "2018-11-26",
        "2019-12-2",
        "2020-11-30",
    ],
}


@pytest.mark.parametrize("holiday", test_dates.keys())
def test_holidays(holiday):
    for test_date in test_dates[holiday]:
        test_date = pd.to_datetime(test_date)
        distance_function = SPECIAL_DATE_FEATURES[holiday]
        assert (
            distance_function(test_date) == 0
        ), "The supplied date should be {} but is not!".format(holiday)


def test_special_date_feature_set_daily():
    date_indices = pd.date_range(
        start="2016-12-24", end="2016-12-31", freq="D"
    )

    reference_features = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ]
    )
    sfs = SpecialDateFeatureSet([CHRISTMAS_EVE, CHRISTMAS_DAY, NEW_YEARS_EVE])
    computed_features = sfs(date_indices)

    assert (
        computed_features == reference_features
    ).all(), "Computed features do not match reference features."


def test_special_date_feature_set_hourly():
    date_indices = pd.date_range(
        start="2016-12-24", end="2016-12-25", freq="H"
    )

    reference_features = np.array(
        [
            [
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                0,
            ],
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
            ],
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
        ]
    )
    sfs = SpecialDateFeatureSet([CHRISTMAS_EVE, CHRISTMAS_DAY, NEW_YEARS_EVE])
    computed_features = sfs(date_indices)

    assert (
        computed_features == reference_features
    ).all(), "Computed features do not match reference features."


def test_special_date_feature_set_daily_squared_exponential():
    date_indices = pd.date_range(
        start="2016-12-24", end="2016-12-29", freq="D"
    )
    reference_features = np.array(
        [
            [
                1.00000e00,
                3.67879e-01,
                1.83156e-02,
                1.23410e-04,
                1.12535e-07,
                0.00000e00,
            ],
            [
                3.67879e-01,
                1.00000e00,
                3.67879e-01,
                1.83156e-02,
                1.23410e-04,
                1.12535e-07,
            ],
        ],
        dtype=float,
    )

    squared_exp_kernel = squared_exponential_kernel(alpha=1.0)
    sfs = SpecialDateFeatureSet(
        [CHRISTMAS_EVE, CHRISTMAS_DAY], squared_exp_kernel
    )
    computed_features = sfs(date_indices)
    np.testing.assert_almost_equal(
        computed_features, reference_features, decimal=6
    )


def test_custom_date_feature_set():

    ref_dates = [
        pd.to_datetime("20191129", format="%Y%m%d"),
        pd.to_datetime("20200101", format="%Y%m%d"),
    ]

    kernel = exponential_kernel(alpha=1.0)

    cfs = CustomDateFeatureSet(ref_dates, kernel)
    sfs = SpecialDateFeatureSet([BLACK_FRIDAY, NEW_YEARS_DAY], kernel)

    date_indices = pd.date_range(
        start=pd.to_datetime("20191101", format="%Y%m%d"),
        end=pd.to_datetime("20200131", format="%Y%m%d"),
        freq="D",
    )

    assert (
        np.sum(cfs(date_indices) - sfs(date_indices).sum(0, keepdims=True))
        == 0
    ), "Features don't match"


def test_custom_holiday_feature_set():

    custom_holidays = [
        Holiday("New Years Day", month=1, day=1),
        Holiday("Christmas Day", month=12, day=25),
    ]

    kernel = exponential_kernel(alpha=1.0)

    cfs = CustomHolidayFeatureSet(custom_holidays, kernel)
    sfs = SpecialDateFeatureSet([NEW_YEARS_DAY, CHRISTMAS_DAY], kernel)

    date_indices = pd.date_range(
        start=pd.to_datetime("20191101", format="%Y%m%d"),
        end=pd.to_datetime("20200131", format="%Y%m%d"),
        freq="D",
    )

    assert (
        np.sum(cfs(date_indices) - sfs(date_indices)) == 0
    ), "Features don't match"
