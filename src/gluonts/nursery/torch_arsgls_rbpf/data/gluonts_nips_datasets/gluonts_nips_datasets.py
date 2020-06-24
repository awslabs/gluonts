import os
import pathlib
from box import Box

from gluonts.dataset.field_names import FieldName
from gluonts.dataset.loader import (
    InferenceDataLoader,
    TrainDataLoader,
    ValidationDataLoader,
)
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
)
from gluonts.dataset.repository.datasets import dataset_recipes
from gluonts.dataset.common import ListDataset
from gluonts.dataset.common import load_datasets
from gluonts.dataset.repository.datasets import get_dataset as get_dataset_gts

import numpy as np
import torch

from models.gls_parameters.issm import CompositeISSM
from experiments.base_config import TimeFeatType
from utils.utils import one_hot


def get_dataset(dataset_name):
    if dataset_name == "wiki2000_nips":
        datasets_root_path = os.path.join(
            os.environ["HOME"], ".mxnet/gluon-ts/datasets"
        )
        dataset_path = os.path.join(datasets_root_path, dataset_name)

        if not os.path.exists(datasets_root_path):
            os.makedirs(datasets_root_path, exist_ok=True)
        if not os.path.exists(dataset_path):
            raise Exception(
                f"you must manually upload the wiki dataset "
                f"and place it to the following folder: {dataset_path}"
            )
        else:
            dataset = load_datasets(
                metadata=pathlib.PurePath(
                    os.path.join(dataset_path, "metadata")
                ),
                train=pathlib.PurePath(os.path.join(dataset_path, "train")),
                test=pathlib.PurePath(os.path.join(dataset_path, "test")),
            )
            if (
                dataset.metadata.freq == "1D"
            ):  # WHY IS WIKI "D" AND THIS IS "1D" ?!
                dataset.metadata.freq = "D"
            return dataset
    else:
        return get_dataset_gts(dataset_name)


def create_input_transform(
    is_train,
    prediction_length,
    past_length,
    use_feat_static_cat,
    use_feat_dynamic_real,
    freq,
    time_features,
    extract_tail_chunks_for_train: bool = False,
):
    SEASON_INDICATORS_FIELD = "seasonal_indicators"
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
                time_features=CompositeISSM.seasonal_features(freq),
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


def create_loaders(
    dataset,
    batch_sizes,
    past_length,
    prediction_length_full,
    prediction_length_rolling,
    num_batches_per_epoch=50,
    num_workers=0,
    extract_tail_chunks_for_train: bool = False,
    val_full_length=True,
):
    """
    The past_length and prediction_length is seriously unintuitive in gluonTS.
    Here is a little summary to make sure it is used it correctly:
    - loader does NOT provide data[-past_length-prediction_length: -prediction_length].
        --> Train set may not include test range. prediction_length does not cut it out.
    - loader instead provides data[-past_length:] and adds prediction_length time_features.
        --> AFTER this loader, we must AGAIN MANUALLY cut out targets[-prediction_length:].

    I had as follows:
    train: past_length=past_length, prediction_length=0
    test: past_length=past_length+n_steps_forecast, prediction_length=0
        && cut out y[-prediction_length:] from batch, but use it for eval.
    Train does not need to forecast
    Test gives the whole thing for features and we cut out the final part for targets.

    Now want to do as they do in the repo.
    for both do: past_length=past_length, prediction_length=prediction_length
    train: does not matter, prediction_length not used
    test: this does not make sense. It still gives you y[-past_length:] ...
    """
    input_transforms = {}
    for name in ["train", "val", "test_full", "test_rolling"]:
        if name == "train":
            prediction_length = 0
            is_train = True
            past_length_ = past_length
        elif name == "val":
            prediction_length = 0
            is_train = False
            past_length_ = past_length + (
                prediction_length_full
                if val_full_length
                else prediction_length_rolling
            )
        elif name == "test_full":
            prediction_length = prediction_length_full
            is_train = False
            past_length_ = past_length
        elif name == "test_rolling":
            prediction_length = prediction_length_rolling
            is_train = False
            past_length_ = past_length
        else:
            raise Exception(f"unknown dataset: {name}")
        input_transforms[name] = create_input_transform(
            is_train=is_train,
            prediction_length=prediction_length,
            past_length=past_length_,
            use_feat_static_cat=True,
            use_feat_dynamic_real=True
            if dataset.metadata.feat_dynamic_real
            else False,
            freq=dataset.metadata.freq,
            time_features=None,
            extract_tail_chunks_for_train=extract_tail_chunks_for_train,
        )

    train_loader = TrainDataLoader(
        dataset=dataset.train,
        transform=input_transforms["train"],
        num_batches_per_epoch=num_batches_per_epoch,
        batch_size=batch_sizes["train"],
        num_workers=num_workers,
        ctx=None,
        dtype=np.float32,
    )
    val_loader = ValidationDataLoader(
        dataset=dataset.train,
        transform=input_transforms["val"],
        batch_size=batch_sizes["val"],
        num_workers=num_workers,
        ctx=None,
        dtype=np.float32,
    )
    test_full_loader = InferenceDataLoader(
        dataset=dataset.test,
        transform=input_transforms["test_full"],
        batch_size=batch_sizes["test_full"],
        num_workers=num_workers,
        ctx=None,
        dtype=np.float32,
    )
    test_rolling_loader = InferenceDataLoader(
        dataset=dataset.test,
        transform=input_transforms["test_rolling"],
        batch_size=batch_sizes["test_rolling"],
        num_workers=num_workers,
        ctx=None,
        dtype=np.float32,
    )

    return (
        train_loader,
        val_loader,
        test_full_loader,
        test_rolling_loader,
        input_transforms,
    )


def transform_gluonts_to_pytorch(
    batch,
    time_features: TimeFeatType,
    cardinalities_feat_static_cat,
    cardinalities_season_indicators,
    bias_y=0.0,
    factor_y=1.0,
    device="cuda",
    dtype=torch.float32,
):
    n_timesteps_targets = batch["past_target"].shape[1]
    n_timesteps_inputs = batch["past_time_feat"].shape[1] + (
        batch["future_time_feat"].shape[1]
        if "future_time_feat" in batch
        else 0
    )

    # Category features
    feat_static_cat = batch["feat_static_cat"].asnumpy()
    feat_static_cat = [
        torch.tensor(feat_static_cat[..., idx], dtype=torch.int64)
        for idx, num_classes in enumerate(cardinalities_feat_static_cat)
    ]
    feat_static_cat = [
        feat[None, ...].repeat((n_timesteps_inputs, 1,))
        for feat in feat_static_cat
    ]
    feat_static_cat = [feat.to(device) for feat in feat_static_cat]

    # Season indicator features
    seasonal_indicators = np.concatenate(
        [batch["past_seasonal_indicators"].asnumpy().transpose(1, 0, 2)]
        + (
            [batch["future_seasonal_indicators"].asnumpy().transpose(1, 0, 2)]
            if "future_seasonal_indicators" in batch
            else []
        ),
        axis=0,
    )
    seasonal_indicators = torch.tensor(
        seasonal_indicators, dtype=torch.int64, device=device
    )
    seasonal_indicator_feat = [
        one_hot(
            labels=seasonal_indicators[:, :, idx],
            num_classes=num_classes,
            is_squeeze=True,
        )
        .to(dtype)
        .to(device)
        for idx, num_classes in enumerate(cardinalities_season_indicators)
    ]
    seasonal_indicator_feat = torch.cat(seasonal_indicator_feat, dim=-1)

    # Time-Features
    time_feat = torch.tensor(
        np.concatenate(
            [batch["past_time_feat"].asnumpy().transpose(1, 0, 2)]
            + (
                [batch["future_time_feat"].asnumpy().transpose(1, 0, 2)]
                if "future_time_feat" in batch
                else []
            ),
            axis=0,
        ),
        dtype=dtype,
        device=device,
    )

    if time_features.value == TimeFeatType.timefeat.value:
        time_features = time_feat
    elif time_features.value == TimeFeatType.seasonal_indicator.value:
        time_features = seasonal_indicator_feat
    elif time_features.value == TimeFeatType.both.value:
        time_features = torch.cat([seasonal_indicator_feat, time_feat], dim=-1)
    elif time_features.value == TimeFeatType.none.value:
        time_features = None
    else:
        raise ValueError(
            f"unexpected value for time_features: {time_features}"
        )

    y = (
        torch.tensor(batch["past_target"].asnumpy())
        .to(dtype)
        .to(device=device)
        .transpose(1, 0)
    )

    # for these experiments we have just 1 type of static feats.
    assert len(feat_static_cat) == 1
    data = Box(
        y=(y - bias_y) / factor_y,
        seasonal_indicators=seasonal_indicators,
        u_time=time_features,
        u_static_cat=feat_static_cat[0],
    )
    return data


def get_cardinalities(dataset, add_trend):
    cardinalities_feat_static_cat = [
        int(feat_static_cat.cardinality)
        for feat_static_cat in dataset.metadata.feat_static_cat
    ]
    cardinalities_season_indicators = [
        issm.n_seasons
        for issm in CompositeISSM.get_from_freq(
            dataset.metadata.freq, add_trend=add_trend
        ).seasonal_issms
    ]
    return Box(
        cardinalities_feat_static_cat=cardinalities_feat_static_cat,
        cardinalities_season_indicators=cardinalities_season_indicators,
    )


if __name__ == "__main__":
    print(f"Available datasets: {list(dataset_recipes.keys())}")

    dataset_names = [
        "exchange_rate_nips",
        "electricity_nips",
        "traffic_nips",
        "solar_nips",
        "wiki-rolling_nips",
    ]

    print(f"considering datasets: {dataset_names}")
    for dataset_name in dataset_names:
        print(f"trying to create loaders for dataset: {dataset_name}")
        add_trend = True
        past_length = 28
        batch_size = 100

        dataset = get_dataset(dataset_name=dataset_name)
        print(f"predicition length: {dataset.metadata.prediction_length}")
        print(
            f"length: train: {len(dataset.train)}, test: {len(dataset.test)}"
        )
        print(
            f"time-series length: {len(next(iter(dataset.train))['target'])}"
        )
        prediction_length_rolling = dataset.metadata.prediction_length
        if dataset.metadata.freq == "H":
            prediction_length_full = 7 * prediction_length_rolling
        elif dataset.metadata.freq in ["B", "D"]:
            prediction_length_full = 5 * prediction_length_rolling
        else:
            raise Exception("unexpected freq")

        (
            train_loader,
            val_loader,
            test_full_loader,
            test_rolling_loader,
            input_transforms,
        ) = create_loaders(
            dataset=dataset,
            batch_sizes={
                "train": batch_size,
                "val": batch_size,
                "test_full": batch_size,
                "test_rolling": batch_size,
            },
            past_length=past_length,
            prediction_length_full=prediction_length_full,
            prediction_length_rolling=prediction_length_rolling,
            num_workers=0,
        )
        batch_train = next(iter(train_loader))
        batch_val = next(iter(val_loader))
        batch_test = next(iter(test_full_loader))

        batch_train = transform_gluonts_to_pytorch(
            batch=batch_train,
            time_features=TimeFeatType.seasonal_indicator,
            cardinalities_feat_static_cat=[
                int(feat_static_cat.cardinality)
                for feat_static_cat in dataset.metadata.feat_static_cat
            ],
            cardinalities_season_indicators=[
                issm.n_seasons
                for issm in CompositeISSM.get_from_freq(
                    dataset.metadata.freq, add_trend=add_trend
                ).seasonal_issms
            ],
        )
        batch_test = transform_gluonts_to_pytorch(
            batch=batch_test,
            time_features=TimeFeatType.seasonal_indicator,
            cardinalities_feat_static_cat=[
                int(feat_static_cat.cardinality)
                for feat_static_cat in dataset.metadata.feat_static_cat
            ],
            cardinalities_season_indicators=[
                issm.n_seasons
                for issm in CompositeISSM.get_from_freq(
                    dataset.metadata.freq, add_trend=add_trend
                ).seasonal_issms
            ],
        )

        # print(batch_train['seasonal_indicators'].shape)
        import matplotlib.pyplot as plt

        # from gluonts.dataset.common import ListDataset
        # data = ListDataset(dataset.train, freq=dataset.metadata.freq).list_data
        # plt.figure()
        # for d in data:
        #     assert d['target'].ndim == 1
        #     plt.plot(d['target'])
        #     plt.title(dataset_name)
        # plt.show()
        # plt.close()

        list_data = ListDataset(
            dataset.train, freq=dataset.metadata.freq
        ).list_data
        # TB
        data_array = np.concatenate(
            [list_data[idx]["target"] for idx in range(len(list_data))],
            axis=0,
        )
        m = data_array.mean()
        std = data_array.std()
        print(m, std)
        # plt.plot((data_array-m)/std)
        # plt.show()
