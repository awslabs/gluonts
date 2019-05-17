# Third-party imports
import numpy as np
import pandas as pd
import pytest

# First-party imports
from gluonts.time_feature.holiday import (
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
    SpecialDateFeatureSet,
    squared_exponential_kernel,
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
}


@pytest.mark.parametrize('holiday', test_dates.keys())
def test_holidays(holiday):
    for test_date in test_dates[holiday]:
        test_date = pd.to_datetime(test_date)
        distance_function = SPECIAL_DATE_FEATURES[holiday]
        assert (
            distance_function(test_date) == 0
        ), "The supplied date should be {} but is not!".format(holiday)


def test_special_date_feature_set_daily():
    date_indices = pd.date_range(
        start="2016-12-24", end="2016-12-31", freq='D'
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
        start="2016-12-24", end="2016-12-25", freq='H'
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
        start="2016-12-24", end="2016-12-29", freq='D'
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
