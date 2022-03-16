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

# If you use this code in your work please cite:
# Multivariate Time Series Forecasting with Latent Graph Inference (https://arxiv.org/abs/2203.03423)

from typing import List
from gluonts.dataset.field_names import FieldName
from gluonts.transform import AddObservedValuesIndicator, InstanceSplitter, ExpectedNumInstanceSampler, TestSplitSampler
from gluonts.torch.batchify import batchify
from gluonts.dataset.loader import TrainDataLoader
from gluonts.itertools import Cached


def wrap_with_dataloader(dataset, batch_size: int, num_batches_per_epoch: int, prediction_length: int,
                         context_length: int) -> TrainDataLoader:
    transformations = get_train_trans(prediction_length, context_length)
    data_loader = TrainDataLoader(
        # We cache the dataset, to make training faster
        Cached(dataset),
        batch_size=batch_size,
        stack_fn=batchify,
        transform=transformations,
        num_batches_per_epoch=num_batches_per_epoch,
    )

    return data_loader


def get_train_trans(prediction_length: int, context_length: int) -> List:
    mask_trans = get_mask_trans()
    train_split_trans = get_train_split_trans(prediction_length, context_length)
    print(type(mask_trans))
    print(type(train_split_trans))
    return mask_trans + train_split_trans


def get_pred_trans(context_length: int, prediction_length: int) -> List:
    mask_trans = get_mask_trans()
    pred_split_trans = get_pred_splitter_trans(context_length, prediction_length)
    return mask_trans + pred_split_trans


def get_mask_trans() -> AddObservedValuesIndicator:
    # Replaces nans in the target field with a dummy value (zero), and adds a field indicating which values were
    # actually observed vs imputed this way.
    mask_unobserved = AddObservedValuesIndicator(
        target_field=FieldName.TARGET,
        output_field=FieldName.OBSERVED_VALUES,
    )
    return mask_unobserved


def get_train_split_trans(prediction_length: int, context_length: int) -> InstanceSplitter:
    training_splitter = InstanceSplitter(
        target_field=FieldName.TARGET,
        is_pad_field=FieldName.IS_PAD,
        start_field=FieldName.START,
        forecast_start_field=FieldName.FORECAST_START,
        instance_sampler=ExpectedNumInstanceSampler(
            num_instances=1,
            min_future=prediction_length,
        ),
        past_length=context_length,
        future_length=prediction_length,
        time_series_fields=[FieldName.OBSERVED_VALUES],
    )
    return training_splitter


def get_pred_splitter_trans(context_length: int, prediction_length: int) -> InstanceSplitter:
    prediction_splitter = InstanceSplitter(
        target_field=FieldName.TARGET,
        is_pad_field=FieldName.IS_PAD,
        start_field=FieldName.START,
        forecast_start_field=FieldName.FORECAST_START,
        instance_sampler=TestSplitSampler(),
        past_length=context_length,
        future_length=prediction_length,
        time_series_fields=[FieldName.OBSERVED_VALUES],
    )
    return prediction_splitter

