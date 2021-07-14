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

import math
import random
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np
import pandas as pd

from gluonts.dataset.artificial.recipe import (
    BinaryHolidays,
    BinaryMarkovChain,
    Constant,
    ForEachCat,
    Lag,
    LinearTrend,
    RandomCat,
    RandomGaussian,
    Stack,
    generate,
    take_as_list,
)
from gluonts.dataset.common import (
    BasicFeatureInfo,
    CategoricalFeatureInfo,
    DataEntry,
    Dataset,
    ListDataset,
    MetaData,
    TrainDatasets,
)
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.stat import (
    DatasetStatistics,
    calculate_dataset_statistics,
)


class DatasetInfo(NamedTuple):
    """
    Information stored on a dataset. When downloading from the repository, the
    dataset repository checks that the obtained version matches the one
    declared in dataset_info/dataset_name.json.
    """

    name: str
    metadata: MetaData
    prediction_length: int
    train_statistics: DatasetStatistics
    test_statistics: DatasetStatistics


class ArtificialDataset:
    """
    Parent class of a dataset that can be generated from code.
    """

    def __init__(self, freq) -> None:
        self.freq = freq

    @property
    def metadata(self) -> MetaData:
        pass

    @property
    def train(self) -> List[DataEntry]:
        pass

    @property
    def test(self) -> List[DataEntry]:
        pass

    # todo return the same type as dataset repo for better usability
    def generate(self) -> TrainDatasets:
        return TrainDatasets(
            metadata=self.metadata,
            train=ListDataset(self.train, self.freq),
            test=ListDataset(self.test, self.freq),
        )


class ConstantDataset(ArtificialDataset):
    def __init__(
        self,
        num_timeseries: int = 10,
        num_steps: int = 30,
        freq: str = "1H",
        start: str = "2000-01-01 00:00:00",
        is_nan: bool = False,  # Generates constant dataset of 0s with explicit NaN missing values
        is_random_constant: bool = False,  # Inserts random constant value for each time series
        is_different_scales: bool = False,  # Generates constants on various scales
        is_piecewise: bool = False,  # Determines whether the time series in the test
        # and train set should have different constant values
        is_noise: bool = False,  # Determines whether to add Gaussian noise to the constant dataset
        is_long: bool = False,  # Determines whether some time series will have very long lengths
        is_short: bool = False,  # Determines whether some time series will have very short lengths
        is_trend: bool = False,  # Determines whether to add linear trends
        num_missing_middle: int = 0,  # Number of missing values in the middle of the time series
        is_promotions: bool = False,  # Determines whether to add promotions to the target time series
        # and to store in metadata
        holidays: Optional[
            List[pd.Timestamp]
        ] = None,  # Determines whether to add holidays to the target time series
        # and to store in metadata
    ) -> None:
        super(ConstantDataset, self).__init__(freq)
        self.num_timeseries = num_timeseries
        self.num_steps = num_steps
        self.num_training_steps = self.num_steps // 10 * 8
        self.prediction_length = self.num_steps - self.num_training_steps
        self.start = start
        self.is_nan = is_nan
        self.is_random_constant = is_random_constant
        self.is_different_scales = is_different_scales
        self.is_piecewise = is_piecewise
        self.is_noise = is_noise
        self.is_long = is_long
        self.is_short = is_short
        self.is_trend = is_trend
        self.num_missing_middle = num_missing_middle
        self.is_promotions = is_promotions
        self.holidays = holidays

    @property
    def metadata(self) -> MetaData:
        metadata = MetaData(
            freq=self.freq,
            feat_static_cat=[
                {
                    "name": "feat_static_cat_000",
                    "cardinality": str(self.num_timeseries),
                }
            ],
            feat_static_real=[{"name": "feat_static_real_000"}],
            prediction_length=self.prediction_length,
        )
        if self.is_promotions or self.holidays:
            metadata = MetaData(
                freq=self.freq,
                feat_static_cat=[
                    {
                        "name": "feat_static_cat_000",
                        "cardinality": str(self.num_timeseries),
                    }
                ],
                feat_static_real=[{"name": "feat_static_real_000"}],
                feat_dynamic_real=[
                    BasicFeatureInfo(name=FieldName.FEAT_DYNAMIC_REAL)
                ],
                prediction_length=self.prediction_length,
            )
        return metadata

    def determine_constant(
        self, index: int, constant: Optional[float] = None, seed: int = 1
    ) -> Optional[float]:
        if self.is_random_constant:
            my_random = random.Random(seed)
            constant = (index + 1) * my_random.random()
        elif self.is_different_scales:
            if index == 0:
                constant = 1e-8
            elif constant is not None:
                constant *= 100
        else:
            constant = float(index)
        return constant

    def compute_data_from_recipe(
        self,
        num_steps: int,
        constant: Optional[float] = None,
        one_to_zero: float = 0.1,
        zero_to_one: float = 0.1,
        scale_features: float = 200,
    ) -> TrainDatasets:
        recipe = []
        recipe_type = Constant(constant)
        if self.is_noise:
            recipe_type += RandomGaussian()  # Use default stddev = 1.0
        if self.is_trend:
            recipe_type += LinearTrend()
        if self.is_promotions:
            recipe.append(
                ("binary_causal", BinaryMarkovChain(one_to_zero, zero_to_one))
            )
            recipe.append(
                (FieldName.FEAT_DYNAMIC_REAL, Stack(["binary_causal"]))
            )
            recipe_type += scale_features * Lag("binary_causal", lag=0)
        if self.holidays:
            timestamp = self.init_date()
            # Compute dates array
            dates = []
            for i in range(num_steps):
                dates.append(timestamp)
                timestamp += 1
            recipe.append(
                ("binary_holidays", BinaryHolidays(dates, self.holidays))
            )
            recipe.append(
                (FieldName.FEAT_DYNAMIC_REAL, Stack(["binary_holidays"]))
            )
            recipe_type += scale_features * Lag("binary_holidays", lag=0)
        recipe.append((FieldName.TARGET, recipe_type))
        max_train_length = num_steps - self.prediction_length
        data = RecipeDataset(
            recipe=recipe,
            metadata=self.metadata,
            max_train_length=max_train_length,
            prediction_length=self.prediction_length,
            num_timeseries=1,  # Add 1 time series at a time in the loop for different constant valus per time series
        )
        generated = data.generate()
        return generated

    def piecewise_constant(self, index: int, num_steps: int) -> List:
        target = []
        for j in range(num_steps):
            if j < self.num_training_steps:
                constant = self.determine_constant(index=index)
            else:
                constant = self.determine_constant(index=index, seed=2)
            target.append(constant)
        return target

    def get_num_steps(
        self,
        index: int,
        num_steps_max: int = 10000,
        long_freq: int = 4,
        num_steps_min: int = 2,
        short_freq: int = 4,
    ) -> int:
        num_steps = self.num_steps
        if self.is_long and index % long_freq == 0:
            num_steps = num_steps_max
        elif self.is_short and index % short_freq == 0:
            num_steps = num_steps_min
        return num_steps

    def init_date(self) -> pd.Timestamp:
        week_dict = {
            0: "MON",
            1: "TUE",
            2: "WED",
            3: "THU",
            4: "FRI",
            5: "SAT",
            6: "SUN",
        }
        timestamp = pd.Timestamp(self.start)
        freq_week_start = self.freq
        if freq_week_start == "W":
            freq_week_start = f"W-{week_dict[timestamp.weekday()]}"
        return pd.Timestamp(self.start, freq=freq_week_start)

    @staticmethod
    def insert_nans_and_zeros(ts_len: int) -> List:
        target = []
        for j in range(ts_len):
            # Place NaNs at even indices. Use convention no NaNs before start date.
            if j != 0 and j % 2 == 0:
                target.append(np.nan)
            # Place zeros at odd indices
            else:
                target.append(0.0)
        return target

    def insert_missing_vals_middle(
        self, ts_len: int, constant: Optional[float]
    ) -> List:
        target = []
        lower_bound = (self.num_training_steps - self.num_missing_middle) // 2
        upper_bound = (self.num_training_steps + self.num_missing_middle) // 2
        num_missing_endpts = math.floor(0.1 * self.num_missing_middle)
        for j in range(ts_len):
            if (
                (0 < j < lower_bound and j % (2 * num_missing_endpts) == 0)
                or (lower_bound <= j < upper_bound)
                or (j >= upper_bound and j % (2 * num_missing_endpts) == 0)
            ):
                val = np.nan
            else:
                val = constant
            target.append(val)
        return target

    def generate_ts(
        self, num_ts_steps: int, is_train: bool = False
    ) -> List[DataEntry]:
        res = []
        constant = None
        for i in range(self.num_timeseries):
            if self.is_nan:
                target = self.insert_nans_and_zeros(num_ts_steps)
            elif self.is_piecewise:
                target = self.piecewise_constant(i, num_ts_steps)
            else:
                constant = self.determine_constant(i, constant)
                if self.num_missing_middle > 0:
                    target = self.insert_missing_vals_middle(
                        num_ts_steps, constant
                    )
                elif (
                    self.is_noise
                    or self.is_trend
                    or self.is_promotions
                    or self.holidays
                ):

                    num_steps = self.get_num_steps(i)
                    generated = self.compute_data_from_recipe(
                        num_steps, constant
                    )
                    if is_train:
                        time_series = generated.train
                    else:
                        assert generated.test is not None
                        time_series = generated.test
                    # returns np array convert to list for consistency
                    target = list(time_series)[0][FieldName.TARGET].tolist()
                else:
                    target = [constant] * num_ts_steps
            ts_data = dict(
                start=self.start,
                target=target,
                item_id=str(i),
                feat_static_cat=[i],
                feat_static_real=[i],
            )
            if self.is_promotions or self.holidays:
                ts_data[FieldName.FEAT_DYNAMIC_REAL] = list(time_series)[0][
                    FieldName.FEAT_DYNAMIC_REAL
                ].tolist()
            res.append(ts_data)
        return res

    @property
    def train(self) -> List[DataEntry]:
        return self.generate_ts(
            num_ts_steps=self.num_training_steps, is_train=True
        )

    @property
    def test(self) -> List[DataEntry]:
        return self.generate_ts(num_ts_steps=self.num_steps)


class ComplexSeasonalTimeSeries(ArtificialDataset):
    """
    Generate sinus time series that ramp up and reach a certain amplitude, and
    level and have additional spikes on each sunday.


    TODO: This could be converted to a RecipeDataset to avoid code duplication.
    """

    def __init__(
        self,
        num_series: int = 100,
        prediction_length: int = 20,
        freq_str: str = "D",
        length_low: int = 30,
        length_high: int = 200,
        min_val: float = -10000,
        max_val: float = 10000,
        is_integer: bool = False,
        proportion_missing_values: float = 0,
        is_noise: bool = True,
        is_scale: bool = True,
        percentage_unique_timestamps: float = 0.07,
        is_out_of_bounds_date: bool = False,
        seasonality: Optional[int] = None,
        clip_values: bool = False,
    ) -> None:
        """
        :param num_series: number of time series generated in the train and
               test set
        :param prediction_length:
        :param freq_str:
        :param length_low: minimum length of a time-series, must be larger than
               prediction_length
        :param length_high: maximum length of a time-series
        :param min_val: min value of a time-series
        :param max_val: max value of a time-series
        :param is_integer: whether the dataset has integers or not
        :param proportion_missing_values:
        :param is_noise: whether to add noise
        :param is_scale: whether to add scale
        :param percentage_unique_timestamps: percentage of random start dates bounded between 0 and 1
        :param is_out_of_bounds_date: determines whether to use very old start dates and start dates far in the future
        :param seasonality: Seasonality of the generated data. If not given uses default seasonality for frequency
        :param clip_values: if True the values will be clipped to [min_val, max_val], otherwise linearly scales them
        """
        assert length_low > prediction_length
        super(ComplexSeasonalTimeSeries, self).__init__(freq_str)
        self.num_series = num_series
        self.prediction_length = prediction_length
        self.length_low = length_low
        self.length_high = length_high
        self.freq_str = freq_str
        self.min_val = min_val
        self.max_val = max_val
        self.is_integer = is_integer
        self.proportion_missing_values = proportion_missing_values
        self.is_noise = is_noise
        self.is_scale = is_scale
        self.percentage_unique_timestamps = percentage_unique_timestamps
        self.is_out_of_bounds_date = is_out_of_bounds_date
        self.seasonality = seasonality
        self.clip_values = clip_values

    @property
    def metadata(self) -> MetaData:
        return MetaData(
            freq=self.freq, prediction_length=self.prediction_length
        )

    def _get_period(self) -> int:
        if self.seasonality is not None:
            return self.seasonality
        if self.freq_str == "M":
            return 24
        elif self.freq_str == "W":
            return 52
        elif self.freq_str == "D":
            return 14
        elif self.freq_str == "H":
            return 24
        elif self.freq_str == "min":
            return 60
        else:
            raise RuntimeError()

    def _get_start(self, index: int, my_random: random.Random) -> str:
        if (
            self.is_out_of_bounds_date and index == 0
        ):  # Add edge case of dates out of normal bounds past date
            start_y, start_m, start_d = (
                1690,
                2,
                7,
            )  # Pandas doesn't allot before 1650
            start_h, start_min = 18, 36
        elif (
            self.is_out_of_bounds_date and index == self.num_series - 1
        ):  # Add edge case of dates out of normal bounds future date
            start_y, start_m, start_d = (
                2030,
                6,
                3,
            )  # Pandas doesn't allot before 1650
            start_h, start_min = 18, 36
        # assume that only 100 * percentage_unique_timestamps of timestamps are unique
        elif my_random.random() < self.percentage_unique_timestamps:
            start_y = my_random.randint(2000, 2018)
            start_m = my_random.randint(1, 12)
            start_d = my_random.randint(1, 28)
            start_h = my_random.randint(0, 23)
            start_min = my_random.randint(0, 59)
        else:
            start_y, start_m, start_d = 2013, 11, 28
            start_h, start_min = 18, 36

        if self.freq_str == "M":
            return "%04.d-%02.d" % (start_y, start_m)
        elif self.freq_str in ["W", "D"]:
            return "%04.d-%02.d-%02.d" % (start_y, start_m, start_d)
        elif self.freq_str == "H":
            return "%04.d-%02.d-%02.d %02.d:00:00" % (
                start_y,
                start_m,
                start_d,
                start_h,
            )
        else:
            return "%04.d-%02.d-%02.d %02.d:%02.d:00" % (
                start_y,
                start_m,
                start_d,
                start_h,
                start_min,
            )

    def _special_time_point_indicator(self, index) -> bool:
        if self.freq_str == "M":
            return index.month == 1
        elif self.freq_str == "W":
            return index.month % 2 == 0
        elif self.freq_str == "D":
            return index.dayofweek == 0
        elif self.freq_str == "H":
            return index.hour == 0
        elif self.freq_str == "min":
            return index.minute % 30 == 0
        else:
            raise RuntimeError(f'Bad freq_str value "{index}"')

    @property
    def train(self) -> List[DataEntry]:
        return [
            dict(
                start=ts[FieldName.START],
                target=ts[FieldName.TARGET][: -self.prediction_length],
                item_id=ts[FieldName.ITEM_ID],
            )
            for ts in self.make_timeseries()
        ]

    @property
    def test(self) -> List[DataEntry]:
        return self.make_timeseries()

    def make_timeseries(self, seed: int = 1) -> List[DataEntry]:
        res = []
        # Fix seed so that the training set is the same
        # as the test set from 0:self.prediction_length for the two independent calls

        def sigmoid(x: np.ndarray) -> np.ndarray:
            return 1.0 / (1.0 + np.exp(-x))

        # Ensure same start dates in test and training set
        my_random = random.Random(seed)
        state = np.random.RandomState(seed)
        for i in range(self.num_series):
            val_range = self.max_val - self.min_val
            length = state.randint(low=self.length_low, high=self.length_high)
            start = self._get_start(i, my_random)
            envelope = sigmoid((np.arange(length) - 20.0) / 10.0)
            level = 0.3 * val_range * (state.random_sample() - 0.5)
            phi = 2 * np.pi * state.random_sample()
            period = self._get_period()
            w = 2 * np.pi / period
            t = np.arange(length)
            idx = pd.date_range(
                start=start, freq=self.freq_str, periods=length
            )
            special_tp_indicator = self._special_time_point_indicator(idx)
            sunday_effect = state.random_sample() * special_tp_indicator
            v = np.sin(w * t + phi) + sunday_effect

            if self.is_scale:
                scale = 0.1 * val_range * state.random_sample()
                v *= scale
            v += level
            if self.is_noise:
                noise_range = 0.02 * val_range * state.random_sample()
                noise = noise_range * state.normal(size=length)
                v += noise
            v = envelope * v
            if self.clip_values:
                np.clip(v, a_min=self.min_val, a_max=self.max_val, out=v)
            else:
                """
                Rather than mapping [v_min, v_max] to [self.min_val, self.max_val] which would lead to
                all the time series having the same min and max, we want to keep the same interval length
                (v_max - v_min). We thus shift the interval [v_min, v_max] in [self.min_val, self.max_val]
                and clip it if needed.
                """
                v_min, v_max = v.min(), v.max()
                p_min, p_max = (
                    max(self.min_val, v_min),
                    min(self.max_val, v_max),
                )
                shifted_min = np.clip(
                    p_min + (p_max - v_max),
                    a_min=self.min_val,
                    a_max=self.max_val,
                )
                shifted_max = np.clip(
                    p_max + (p_min - v_min),
                    a_min=self.min_val,
                    a_max=self.max_val,
                )
                v = shifted_min + (shifted_max - shifted_min) * (v - v_min) / (
                    v_max - v_min
                )

            if self.is_integer:
                np.clip(
                    v,
                    a_min=np.ceil(self.min_val),
                    a_max=np.floor(self.max_val),
                    out=v,
                )
                v = np.round(v).astype(int)
            v = list(v.tolist())
            if self.proportion_missing_values > 0:
                assert (
                    self.proportion_missing_values < 1.0
                ), "Please chose a number 0 < x < 1.0"
                idx = np.arange(len(v))
                state.shuffle(idx)
                num_missing_values = (
                    int(len(v) * self.proportion_missing_values) + 1
                )  # Add one in case this gets zero
                missing_idx = idx[:num_missing_values]
                for j in missing_idx:
                    # Using convention that there are no missing values before the start date.
                    if j != 0:
                        v[j] = None if state.rand() < 0.5 else "NaN"
            res.append(
                dict(
                    start=pd.Timestamp(start, freq=self.freq_str),
                    target=np.array(v),
                    item_id=i,
                )
            )
        return res


class RecipeDataset(ArtificialDataset):
    """Synthetic data set generated by providing a recipe.

    A recipe is either a (non-deterministic) function

        f(length: int, global_state: dict) -> dict

    or list of (field, function) tuples of the form

        (field: str, f(data: dict, length: int, global_state: dict) -> dict)

    which is processed sequentially, with data initially set to {},
    and each entry updating data[field] to the output of the function
    call.
    """

    def __init__(
        self,
        recipe: Union[
            Callable, Dict[str, Callable], List[Tuple[str, Callable]]
        ],
        metadata: MetaData,
        max_train_length: int,
        prediction_length: int,
        num_timeseries: int,
        trim_length_fun=lambda x, **kwargs: 0,
        data_start=pd.Timestamp("2014-01-01"),
    ) -> None:
        """

        :param recipe: The recipe to generate from (see class docstring)
        :param metadata: The metadata to be included in the dataset
        :param max_train_length: The maximum length of a training time series.
        :param prediction_length: The length of the prediction range
        :param num_timeseries: Number of time series to generate
        :param trim_length_fun: Callable f(x: int) -> int returning the
               (shortened) training length
        :param data_start: Start date for the data set
        """
        super().__init__(freq=metadata.freq)

        self.recipe = recipe
        self._metadata = metadata
        self.max_train_length = max_train_length
        self.prediction_length = prediction_length
        self.trim_length_fun = trim_length_fun
        self.num_timeseries = num_timeseries
        self.data_start = pd.Timestamp(data_start, freq=self._metadata.freq)

    @property
    def metadata(self) -> MetaData:
        return self._metadata

    def dataset_info(self, train_ds: Dataset, test_ds: Dataset) -> DatasetInfo:
        return DatasetInfo(
            name=f"RecipeDataset({repr(self.recipe)})",
            metadata=self.metadata,
            prediction_length=self.prediction_length,
            train_statistics=calculate_dataset_statistics(train_ds),
            test_statistics=calculate_dataset_statistics(test_ds),
        )

    @staticmethod
    def trim_ts_item_end(x: DataEntry, length: int) -> DataEntry:
        """Trim a DataEntry into a training range, by removing
        the last prediction_length time points from the target and dynamic
        features."""
        y = dict(
            item_id=x[FieldName.ITEM_ID],
            start=x[FieldName.START],
            target=x[FieldName.TARGET][:-length],
        )

        if FieldName.FEAT_DYNAMIC_CAT in x:
            y[FieldName.FEAT_DYNAMIC_CAT] = x[FieldName.FEAT_DYNAMIC_CAT][
                :, :-length
            ]
        if FieldName.FEAT_DYNAMIC_REAL in x:
            y[FieldName.FEAT_DYNAMIC_REAL] = x[FieldName.FEAT_DYNAMIC_REAL][
                :, :-length
            ]
        return y

    @staticmethod
    def trim_ts_item_front(x: DataEntry, length: int) -> DataEntry:
        """Trim a DataEntry into a training range, by removing
        the first offset_front time points from the target and dynamic
        features."""
        assert length <= len(x[FieldName.TARGET])

        y = dict(
            item_id=x[FieldName.ITEM_ID],
            start=x[FieldName.START] + length * x[FieldName.START].freq,
            target=x[FieldName.TARGET][length:],
        )

        if FieldName.FEAT_DYNAMIC_CAT in x:
            y[FieldName.FEAT_DYNAMIC_CAT] = x[FieldName.FEAT_DYNAMIC_CAT][
                :, length:
            ]
        if FieldName.FEAT_DYNAMIC_REAL in x:
            y[FieldName.FEAT_DYNAMIC_REAL] = x[FieldName.FEAT_DYNAMIC_REAL][
                :, length:
            ]
        return y

    def generate(self) -> TrainDatasets:
        metadata = self.metadata
        data_it = generate(
            length=self.max_train_length + self.prediction_length,
            recipe=self.recipe,
            start=self.data_start,
        )
        full_length_data = take_as_list(data_it, self.num_timeseries)

        test_data = [
            RecipeDataset.trim_ts_item_front(
                x, self.trim_length_fun(x, train_length=self.max_train_length)
            )
            for x in full_length_data
        ]
        train_data = [
            RecipeDataset.trim_ts_item_end(x, self.prediction_length)
            for x in test_data
        ]
        return TrainDatasets(
            metadata=metadata,
            train=ListDataset(train_data, metadata.freq),
            test=ListDataset(test_data, metadata.freq),
        )


def default_synthetic() -> Tuple[DatasetInfo, Dataset, Dataset]:

    recipe = [
        (FieldName.TARGET, LinearTrend() + RandomGaussian()),
        (FieldName.FEAT_STATIC_CAT, RandomCat([10])),
        (
            FieldName.FEAT_STATIC_REAL,
            ForEachCat(RandomGaussian(1, (10,)), FieldName.FEAT_STATIC_CAT)
            + RandomGaussian(0.1, (10,)),
        ),
    ]

    data = RecipeDataset(
        recipe=recipe,
        metadata=MetaData(
            freq="D",
            feat_static_real=[
                BasicFeatureInfo(name=FieldName.FEAT_STATIC_REAL)
            ],
            feat_static_cat=[
                CategoricalFeatureInfo(
                    name=FieldName.FEAT_STATIC_CAT, cardinality=10
                )
            ],
            feat_dynamic_real=[
                BasicFeatureInfo(name=FieldName.FEAT_DYNAMIC_REAL)
            ],
        ),
        max_train_length=20,
        prediction_length=10,
        num_timeseries=10,
        trim_length_fun=lambda x, **kwargs: np.minimum(
            int(np.random.geometric(1 / (kwargs["train_length"] / 2))),
            kwargs["train_length"],
        ),
    )

    generated = data.generate()
    assert generated.test is not None
    info = data.dataset_info(generated.train, generated.test)

    return info, generated.train, generated.test


def constant_dataset() -> Tuple[DatasetInfo, Dataset, Dataset]:
    metadata = MetaData(
        freq="1H",
        feat_static_cat=[
            CategoricalFeatureInfo(
                name="feat_static_cat_000", cardinality="10"
            )
        ],
        feat_static_real=[BasicFeatureInfo(name="feat_static_real_000")],
    )

    start_date = "2000-01-01 00:00:00"

    train_ds = ListDataset(
        data_iter=[
            {
                FieldName.ITEM_ID: str(i),
                FieldName.START: start_date,
                FieldName.TARGET: [float(i)] * 24,
                FieldName.FEAT_STATIC_CAT: [i],
                FieldName.FEAT_STATIC_REAL: [float(i)],
            }
            for i in range(10)
        ],
        freq=metadata.freq,
    )

    test_ds = ListDataset(
        data_iter=[
            {
                FieldName.ITEM_ID: str(i),
                FieldName.START: start_date,
                FieldName.TARGET: [float(i)] * 30,
                FieldName.FEAT_STATIC_CAT: [i],
                FieldName.FEAT_STATIC_REAL: [float(i)],
            }
            for i in range(10)
        ],
        freq=metadata.freq,
    )

    info = DatasetInfo(
        name="constant_dataset",
        metadata=metadata,
        prediction_length=2,
        train_statistics=calculate_dataset_statistics(train_ds),
        test_statistics=calculate_dataset_statistics(test_ds),
    )

    return info, train_ds, test_ds
