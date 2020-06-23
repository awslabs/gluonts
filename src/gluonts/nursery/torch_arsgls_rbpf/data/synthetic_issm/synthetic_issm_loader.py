import os
import pathlib
import math
from box import Box
import numpy as np
import torch

from gluonts.dataset.common import FileDataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.loader import InferenceDataLoader, TrainDataLoader
from gluonts.time_feature import time_features_from_frequency_str
from gluonts.model.deepstate.issm import CompositeISSM
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

import consts
from experiments.base_config import TimeFeatType
from utils.utils import one_hot


def create_input_transform(
        is_train,
        prediction_length,
        past_length,
        use_feat_static_cat,
        use_feat_dynamic_real,
        freq,
        time_features,
        extract_tail_chunks_for_train: bool = False
):
    SEASON_INDICATORS_FIELD = "seasonal_indicators"
    remove_field_names = [
        FieldName.FEAT_DYNAMIC_CAT,
        FieldName.FEAT_STATIC_REAL,
    ]
    if not use_feat_dynamic_real:
        remove_field_names.append(FieldName.FEAT_DYNAMIC_REAL)

    time_features = time_features if time_features is not None \
        else time_features_from_frequency_str(freq)

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
                instance_sampler=ExpectedNumInstanceSampler(
                    num_instances=1) if (
                        is_train and not extract_tail_chunks_for_train) else TestSplitSampler(),
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


def gluonts_batch_to_train_pytorch(batch, device, dtype, dims,
                                   time_features: TimeFeatType):
    n_timesteps = batch["past_target"].shape[1]
    categories = np.expand_dims(
        batch["feat_static_cat"].asnumpy()[..., 0], axis=0
    ).repeat(n_timesteps, axis=0)
    time_feat = torch.tensor(batch['past_time_feat'].asnumpy(),
                             dtype=torch.float32, device=device).transpose(0, 1)
    assert categories.shape == (n_timesteps, dims.batch)
    categories = one_hot(labels=torch.tensor(categories).to(torch.int64),
                         num_classes=5).to(dtype).to(device=device)

    seasonal_indicators = torch.tensor(
        batch["past_seasonal_indicators"][..., 0].asnumpy()
    ).transpose(0, 1).to(torch.int64).to(device)
    seasonal_indicator_feat = one_hot(
        labels=seasonal_indicators[:, :],
        num_classes=7,
        is_squeeze=True,
    ).to(dtype).to(device)

    if time_features.value == TimeFeatType.timefeat.value:
        time_features = time_feat
    elif time_features.value == TimeFeatType.seasonal_indicator.value:
        time_features = seasonal_indicator_feat
    elif time_features.value == TimeFeatType.both.value:
        time_features = torch.cat([time_feat, seasonal_indicator_feat], dim=-1)
    elif time_features.value == TimeFeatType.none.value:
        time_features = None
    else:
        raise ValueError(f"unexpected value for time_features: {time_features}")

    y = torch.tensor(
        batch["past_target"].asnumpy()
    ).to(dtype).to(device=device).transpose(1, 0)

    data = Box(
        y=y,
        seasonal_indicators=seasonal_indicators,
        u_time=time_features,
        u_static_cat=categories,
    )
    return data


def gluonts_batch_to_forecast_pytorch(*args, n_steps_forecast, **kwargs):
    data = gluonts_batch_to_train_pytorch(*args, **kwargs)
    data.y = data.y[:-n_steps_forecast]  # remove prediction range.
    return data


def create_trainset_loader(n_data_per_group, batch_size,
                           past_length=4 * 7, prediction_length=2 * 7,
                           n_groups=5, dataset_name="synthetic_issm"):
    input_transform = create_input_transform(
        prediction_length=prediction_length,
        past_length=past_length,
        use_feat_static_cat=True,
        use_feat_dynamic_real=False,
        freq="D",
        time_features=None,
        is_train=True,
        extract_tail_chunks_for_train=True,
    )
    dataset = get_dataset(subset="train", dataset_name=dataset_name,
                          n_data_per_group=n_data_per_group)
    dataloader = TrainDataLoader(
        dataset=dataset,
        transform=input_transform,
        num_batches_per_epoch=math.ceil(
            n_data_per_group * n_groups / batch_size),
        batch_size=batch_size,
        ctx=None,  # mx.context.cpu(),
        dtype=np.float32,
    )
    return dataloader


def create_inference_loader(n_data_per_group, batch_size,
                            past_length=4 * 7, prediction_length=2 * 7,
                            dataset_name="synthetic_issm"):
    input_transform = create_input_transform(
        prediction_length=prediction_length,
        past_length=past_length,
        use_feat_static_cat=True,
        use_feat_dynamic_real=False,
        freq="D",
        time_features=None,
        is_train=False,
    )
    dataset = get_dataset(subset="test", dataset_name=dataset_name,
                          n_data_per_group=n_data_per_group)
    dataloader = InferenceDataLoader(
        dataset=dataset,
        transform=input_transform,
        batch_size=batch_size,
        ctx=None,
        dtype=np.float32,
    )
    return dataloader


def get_dataset(dataset_name, subset, n_data_per_group):
    assert subset in ["train", "test"]
    folder_path = os.path.join(
        consts.data_dir, dataset_name, str(n_data_per_group), subset)
    assert os.path.exists(folder_path), f"path does not exist. " \
                                        f"bad n_data_per_group? " \
                                        f"Got: {n_data_per_group}"
    path = pathlib.PurePath(folder_path)
    dataset = FileDataset(path, freq="D")
    return dataset


if __name__ == "__main__":
    # TODO: make test and move to test folder.

    dataloader = create_trainset_loader(n_data_per_group=40, batch_size=100)
    for idx, batch in enumerate(dataloader):
        batch = gluonts_batch_to_train_pytorch(batch=batch, device="cuda")
        # print(idx)
    assert isinstance(batch, dict)
