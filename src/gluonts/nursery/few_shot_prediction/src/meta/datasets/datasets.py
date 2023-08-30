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

import itertools
from sklearn.model_selection import train_test_split
from math import isclose
import numpy as np


FREQ_M1_M3 = ["monthly", "quarterly", "yearly"]
FREQ_M4 = ["daily", "weekly", "monthly", "quarterly", "yearly"]

CATEGORIES_M1 = [
    "micro1",
    "micro2",
    "micro3",
    "indust",
    "macro1",
    "macro2",
    "demogr",
]

CATEGORIES_M3_M4 = [
    "micro",
    "industry",
    "macro",
    "finance",
    "demographic",
]


M1 = [
    "m1_" + freq + "_" + cat
    for freq, cat in itertools.product(FREQ_M1_M3, CATEGORIES_M1)
]
M3 = [
    "m3_" + freq + "_" + cat
    for freq, cat in itertools.product(FREQ_M1_M3, CATEGORIES_M3_M4)
]
M4 = [
    "m4_" + freq + "_" + cat
    for freq, cat in itertools.product(FREQ_M4, CATEGORIES_M3_M4)
]

DATASETS_SINGLE = [
    # "exchange_rate",
    "exchange_rate_nips",
    # "solar-energy",
    "solar_nips",
    # "electricity",
    "electricity_nips",
    # "traffic",
    "traffic_nips",
    "wiki-rolling_nips",  # is a very challenging dataset
    "taxi_30min",
    "kaggle_web_traffic_without_missing",
    "kaggle_web_traffic_weekly",
    "nn5_daily_without_missing",  # http://www.neural-forecasting-competition.com/NN5/  (ATM cash withdrawals)
    "nn5_weekly",
    "tourism_monthly",  # no need to split, https://robjhyndman.com/papers/forecompijf.pdf
    "tourism_quarterly",
    "tourism_yearly",
    "cif_2016",  # https://irafm.osu.cz/cif/main.php?c=Static&page=download ,
    # https://zenodo.org/record/3904073#.YeE6ZMYo8UE , 72 time series, 24 real, 48 artificial, all monthly, banking domain
    "london_smart_meters_without_missing",
    "wind_farms_without_missing",
    # "car_parts_without_missing",  # intermittent time series
    # "dominick",  # This dataset contains 115704 weekly time series representing the profit of individual stock keeping units from a retailer.
    # https://www.chicagobooth.edu/research/kilts/datasets/dominicks
    # dominick has intermittency and other problems
    # TODO: Could this be clustered? (as step 2, it is already from one category only)
    "fred_md",  # https://s3.amazonaws.com/real.stlouisfed.org/wp/2015/2015-012.pdf , too mixed up?
    "pedestrian_counts",  # hourly pedestrian counts captured from 66 sensors in Melbourne city starting from May 2009
    "hospital",  # https://zenodo.org/record/4656014#.YeFBT8Yo8UE
    # This dataset contains 767 monthly time series that represent the patient counts related to medical products from January 2000 to December 2006. It was extracted from R expsmooth package.
    "covid_deaths",  # https://zenodo.org/record/4656009#.YeFBq8Yo8UE
    # This dataset contains 266 daily time series that represent the COVID-19 deaths in a set of countries and states from 22/01/2020 to 20/08/2020. It was extracted from the Johns Hopkins repository.
    "kdd_cup_2018_without_missing",  # https://zenodo.org/record/4656756#.YeFB7cYo8UE
    # This dataset was used in the KDD Cup 2018 forecasting competition.
    # It contains long hourly time series representing the air quality levels
    # in 59 stations in 2 cities: Beijing (35 stations) and London (24 stations) from 01/01/2017 to 31/03/2018.
    # TODO: could be clustered by cities
    "weather",  # https://zenodo.org/record/4654822#.YeFCXcYo8UE
    # 3010 daily time series representing the variations of four weather variables: rain, mintemp, maxtemp and solar radiation, measured at the weather stations in Australia
    # TODO: could be clustered by variable (as step 2, it is already from one category only)
    # "m5",  # intermittent time series, see https://mofc.unic.ac.cy/m5-competition/, Walmart product data, could be splitted by product or store
]

DATASETS_FULL = DATASETS_SINGLE + M1 + M3 + M4

# The excluded datasets contain only time series that are shorter than 3 predictions lengths.
# In the current setting we filter these time series.
DATASETS_FILTERED = [
    ds
    for ds in DATASETS_FULL
    if ds not in ["m1_yearly_macro2", "m3_yearly_micro", "m3_yearly_macro"]
]


def sample_datasets(
    n_folds: int,
    train_size: float = 0.7,
    val_size: float = 0.2,
    test_size: float = 0.1,
    seed=42,
):
    assert isclose(
        train_size + val_size + test_size, 1.0
    ), "sizes need to add up to 1"
    random_state = np.random.RandomState(seed)
    folds = []
    for _ in range(n_folds):
        train_split, test_split = train_test_split(
            DATASETS_FILTERED,
            train_size=train_size,
            random_state=random_state.randint(low=0, high=10000),
        )
        rest = 1 - train_size
        val_split, test_split = train_test_split(
            test_split,
            test_size=test_size / rest,
            random_state=random_state.randint(low=0, high=10000),
        )
        folds.append((train_split, val_split, test_split))
        assert not any(
            [
                set(train_split) & (set(val_split)),
                set(train_split) & set(test_split),
                set(val_split) & set(test_split),
            ]
        ), "Splits should not intersect!"
    return folds
