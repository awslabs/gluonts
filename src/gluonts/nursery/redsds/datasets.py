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

import torch
import numpy as np
from typing import Iterable
from pathlib import Path

# GluonTS Dataset(s) Imports
from pandas.tseries.frequencies import to_offset

from gluonts.dataset.field_names import FieldName
from gluonts.dataset.common import load_datasets
from gluonts.time_feature import time_features_from_frequency_str
from gluonts.transform import (
    AddObservedValuesIndicator,
    AddAgeFeature,
    AddTimeFeatures,
    AsNumpyArray,
    Chain,
    CanonicalInstanceSplitter,
    ExpandDimArray,
    RemoveFields,
    SetField,
    TestSplitSampler,
    ExpectedNumInstanceSampler,
    VstackFeatures,
    TransformedDataset,
    SelectFields,
)
from gluonts.time_feature import (
    MinuteOfHour,
    HourOfDay,
    DayOfWeek,
    WeekOfYear,
    MonthOfYear,
)

from gluonts.dataset.loader import (
    TrainDataLoader,
    ValidationDataLoader,
)

from gluonts.torch.batchify import batchify

from gluonts.dataset.repository.datasets import get_dataset as get_dataset_gts


SEASON_INDICATORS_FIELD = "seasonal_indicators"


class CachedIterable:
    def __init__(self, iterable: Iterable) -> None:
        self.iterable = iterable
        self.cache = None

    def __iter__(self):
        if self.cache is None:
            self.cache = []
            for element in self.iterable:
                yield element
                self.cache.append(element)
        else:
            yield from self.cache


# Bouncing Ball
class BouncingBallDataset(torch.utils.data.Dataset):
    def __init__(self, path="./data/bouncing_ball.npz"):
        npz = np.load(path)
        self.data_y = npz["y"].astype(np.float32)
        self.data_z = npz["z"].astype(np.int32)

    def __getitem__(self, i):
        return self.data_y[i], self.data_z[i]

    def __len__(self):
        return self.data_y.shape[0]


# Three Mode System
class ThreeModeSystemDataset(torch.utils.data.Dataset):
    def __init__(self, path="./data/3modesystem.npz"):
        npz = np.load(path)
        self.data_y = npz["y"].astype(np.float32)
        self.data_z = npz["z"].astype(np.int32)

    def __getitem__(self, i):
        return self.data_y[i], self.data_z[i]

    def __len__(self):
        return self.data_y.shape[0]


# Bee Dataset
class BeeDataset(torch.utils.data.Dataset):
    def __init__(self, path="./data/bee.npz"):
        npz = np.load(path)
        self.data_y = npz["y"].astype(np.float32)
        self.data_z = npz["z"].astype(np.int32)

    def __getitem__(self, i):
        return self.data_y[i], self.data_z[i]

    def __len__(self):
        return self.data_y.shape[0]


# GluonTS Univariate Datasets


def get_wiki2000_nips(train_path, test_path):
    return load_datasets(
        metadata=train_path / "wiki2000_nips",
        train=train_path / "wiki2000_nips/train",
        test=test_path / "wiki2000_nips/test",
    )


def create_input_transform(
    is_train,
    prediction_length,
    past_length,
    use_feat_static_cat=True,
    use_feat_dynamic_real=False,
    freq="H",
    time_features=None,
    extract_tail_chunks_for_train=False,
):
    def seasonal_features(freq):
        offset = to_offset(freq)
        if offset.name == "M":
            return [MonthOfYear(normalized=False)]
        elif offset.name == "W-SUN":
            return [WeekOfYear(normalized=False)]
        elif offset.name == "D":
            return [DayOfWeek(normalized=False)]
        elif offset.name == "B":
            return [DayOfWeek(normalized=False)]
        elif offset.name == "H":
            return [HourOfDay(normalized=False), DayOfWeek(normalized=False)]
        elif offset.name == "T":
            return [
                MinuteOfHour(normalized=False),
                HourOfDay(normalized=False),
            ]
        else:
            RuntimeError(f"Unsupported frequency {offset.name}")

        return []

    remove_field_names = [
        FieldName.FEAT_DYNAMIC_CAT,
        FieldName.FEAT_STATIC_REAL,
    ]
    if not use_feat_dynamic_real:
        remove_field_names.append(FieldName.FEAT_DYNAMIC_REAL)

    time_features = (
        time_features
        if time_features is not None
        else time_features_from_frequency_str(freq)
    )

    transform = Chain(
        [RemoveFields(field_names=remove_field_names)]
        + (
            [SetField(output_field=FieldName.FEAT_STATIC_CAT, value=[0.0])]
            if not use_feat_static_cat
            else []
        )
        + [
            AsNumpyArray(field=FieldName.FEAT_STATIC_CAT, expected_ndim=1),
            AsNumpyArray(field=FieldName.TARGET, expected_ndim=1),
            # gives target the (1, T) layout
            ExpandDimArray(field=FieldName.TARGET, axis=0),
            AddObservedValuesIndicator(
                target_field=FieldName.TARGET,
                output_field=FieldName.OBSERVED_VALUES,
            ),
            # Unnormalized seasonal features
            AddTimeFeatures(
                time_features=seasonal_features(freq),
                pred_length=prediction_length,
                start_field=FieldName.START,
                target_field=FieldName.TARGET,
                output_field=SEASON_INDICATORS_FIELD,
            ),
            AddTimeFeatures(
                start_field=FieldName.START,
                target_field=FieldName.TARGET,
                output_field=FieldName.FEAT_TIME,
                time_features=time_features,
                pred_length=prediction_length,
            ),
            AddAgeFeature(
                target_field=FieldName.TARGET,
                output_field=FieldName.FEAT_AGE,
                pred_length=prediction_length,
                log_scale=True,
            ),
            VstackFeatures(
                output_field=FieldName.FEAT_TIME,
                input_fields=[FieldName.FEAT_TIME, FieldName.FEAT_AGE]
                + (
                    [FieldName.FEAT_DYNAMIC_REAL]
                    if use_feat_dynamic_real
                    else []
                ),
            ),
            CanonicalInstanceSplitter(
                target_field=FieldName.TARGET,
                is_pad_field=FieldName.IS_PAD,
                start_field=FieldName.START,
                forecast_start_field=FieldName.FORECAST_START,
                instance_sampler=ExpectedNumInstanceSampler(num_instances=1),
                time_series_fields=[
                    FieldName.FEAT_TIME,
                    SEASON_INDICATORS_FIELD,
                    FieldName.OBSERVED_VALUES,
                ],
                allow_target_padding=True,
                instance_length=past_length,
                use_prediction_features=True,
                prediction_length=prediction_length,
            )
            if (is_train and not extract_tail_chunks_for_train)
            else CanonicalInstanceSplitter(
                target_field=FieldName.TARGET,
                is_pad_field=FieldName.IS_PAD,
                start_field=FieldName.START,
                forecast_start_field=FieldName.FORECAST_START,
                instance_sampler=TestSplitSampler(),
                time_series_fields=[
                    FieldName.FEAT_TIME,
                    SEASON_INDICATORS_FIELD,
                    FieldName.OBSERVED_VALUES,
                ],
                allow_target_padding=True,
                instance_length=past_length,
                use_prediction_features=True,
                prediction_length=prediction_length,
            ),
        ]
    )
    return transform


def get_cardinalities(dataset):
    cardinalities_feat_static_cat = [
        int(feat_static_cat.cardinality)
        for feat_static_cat in dataset.metadata.feat_static_cat
    ]

    def get_from_freq(freq):
        offset = to_offset(freq)

        if offset.name == "M":
            cardinalities = [12]  # month-of-year seasonality
        elif offset.name == "W-SUN":
            cardinalities = [53]  # week-of-year seasonality
        elif offset.name == "D":
            cardinalities = [7]  # day-of-week seasonality
        elif offset.name == "B":
            cardinalities = [7]  # day-of-week seasonality
        elif offset.name == "H":
            cardinalities = [
                24,  # hour-of-day seasonality
                7,  # day-of-week seasonality
            ]
        elif offset.name == "T":
            cardinalities = [
                60,  # minute-of-hour seasonality
                24,  # hour-of-day seasonality
            ]
        else:
            ValueError(f"Unsupported frequency {offset.name}")
        return cardinalities

    cardinalities_season_indicators = get_from_freq(dataset.metadata.freq)
    return dict(
        cardinalities_feat_static_cat=cardinalities_feat_static_cat,
        cardinalities_season_indicators=cardinalities_season_indicators,
    )


class GTSUnivariateDataset(torch.utils.data.IterableDataset):
    available_datasets = {
        "exchange_rate_nips",
        "electricity_nips",
        "traffic_nips",
        "solar_nips",
        "wiki2000_nips",
    }
    past_lengths = {
        "exchange_rate_nips": 4 * 31,
        "electricity_nips": 2 * 168,
        "traffic_nips": 2 * 168,
        "solar_nips": 2 * 168,
        "wiki2000_nips": 4 * 31,
    }

    def __init__(
        self,
        dataset_name,
        time_feat_type="time",
        num_batches_per_epoch=250,
        batch_size=50,
        mode="train",
        float_dtype=torch.float32,
        train_path=Path("./data"),
        test_path=Path("./data"),
    ):
        assert dataset_name in GTSUnivariateDataset.available_datasets, (
            f"Unknown dataset! {dataset_name} not in"
            f" {GTSUnivariateDataset.available_datasets}"
        )
        if dataset_name == "wiki2000_nips":
            dataset = get_wiki2000_nips(train_path, test_path)
        else:
            dataset = get_dataset_gts(dataset_name)
            # ^regenerate=True if needed
        cardinalities = get_cardinalities(dataset)
        self.time_feat_type = time_feat_type
        self.freq = dataset.metadata.freq
        context_length = GTSUnivariateDataset.past_lengths[dataset_name]
        prediction_length_rolling = dataset.metadata.prediction_length
        if self.freq == "H":
            prediction_length_full = 7 * 24  # 7 * prediction_length_rolling
        elif self.freq in {"B", "D", "1D"}:
            prediction_length_full = 5 * prediction_length_rolling
        else:
            raise ValueError(f"Unknown freq {self.freq}.")

        if mode == "train":
            train = True
            self.context_length = context_length
            self.gluonts_dataset = dataset.train
        elif mode == "val":
            train = False
            self.context_length = context_length + prediction_length_full
            self.gluonts_dataset = dataset.train
        elif mode == "test":
            train = False
            self.context_length = context_length + prediction_length_full
            self.gluonts_dataset = dataset.test

        input_transform = create_input_transform(
            is_train=train,
            prediction_length=0,
            past_length=self.context_length,
            use_feat_static_cat=True,
            use_feat_dynamic_real=False,
            freq=self.freq,
            time_features=None,
        )

        self.infinite_iter = False

        if mode == "train":
            fields_to_keep = [
                FieldName.TARGET,
                FieldName.START,
                SEASON_INDICATORS_FIELD,
                FieldName.FEAT_STATIC_CAT,
                FieldName.FEAT_TIME,
                FieldName.OBSERVED_VALUES,
            ]

            pre_split_tfs = Chain(
                input_transform.transformations[:-1]
                + [SelectFields(fields_to_keep)]
            )
            splitter = input_transform.transformations[-1]
            assert isinstance(splitter, CanonicalInstanceSplitter)

            transformed_dataset = TransformedDataset(
                dataset.train, pre_split_tfs
            )
            transformed_dataset = CachedIterable(transformed_dataset)

            gts_loader = TrainDataLoader(
                dataset=transformed_dataset,
                transform=splitter,
                batch_size=batch_size,
                stack_fn=batchify,
                num_batches_per_epoch=num_batches_per_epoch,
                num_workers=1,
            )
            self.infinite_iter = True
        elif mode == "val":
            gts_loader = ValidationDataLoader(
                dataset=dataset.train,
                transform=input_transform,
                batch_size=batch_size,
                stack_fn=batchify,
                num_workers=1,
            )
        elif mode == "test":
            gts_loader = ValidationDataLoader(
                dataset=dataset.test,
                transform=input_transform,
                batch_size=batch_size,
                stack_fn=batchify,
                num_workers=1,
            )
        else:
            raise ValueError(f"Unknown mode {mode}.")
        self.data_loader = gts_loader

        # Data keys

        self.float_dtype = float_dtype
        self.int_dtype = torch.int64

        self._all_data_keys = [
            "feat_static_cat",
            "past_target",
            "past_seasonal_indicators",
            "past_time_feat",
            "future_target",
            "future_seasonal_indicators",
            "future_time_feat",
            "past_observed_values",
        ]
        self._static_data_keys = [
            "feat_static_cat",
        ]
        self._int_data_keys = [
            "past_seasonal_indicators",
            "future_seasonal_indicators",
        ]

        # Metadata

        n_staticfeat = sum(cardinalities["cardinalities_feat_static_cat"])
        if time_feat_type == "seasonal":
            n_timefeat = sum(cardinalities["cardinalities_season_indicators"])
        elif time_feat_type == "time":
            n_timefeat = (
                len(cardinalities["cardinalities_season_indicators"]) + 3
            )
        elif time_feat_type == "both":
            n_timefeat = (
                sum(cardinalities["cardinalities_season_indicators"])
                + len(cardinalities["cardinalities_season_indicators"])
                + 3
            )
        elif time_feat_type == "none":
            n_timefeat = 0
        else:
            raise ValueError(f"Unknown time_feat_type {time_feat_type}")

        # TODO: Implement other features, if needed.
        if time_feat_type != "time":
            raise NotImplementedError(
                "Only time_feat_type = time is implemented currently"
            )

        self.metadata = dict(
            n_staticfeat=n_staticfeat,
            n_timefeat=n_timefeat,
            freq=self.freq,
            context_length=context_length,
            prediction_length=prediction_length_full,
        )

    def __iter__(self):
        def data_gen():
            epoch = 0
            while epoch == 0 or self.infinite_iter:
                epoch += 1
                for batch in self.data_loader:
                    relevant_data = {
                        k: batch[k] for k in self._all_data_keys if k in batch
                    }
                    torch_batch = {
                        k: v.type(
                            self.int_dtype
                            if k in self._int_data_keys
                            else self.float_dtype
                        )
                        for k, v in relevant_data.items()
                    }
                    yield torch_batch

        return data_gen()
