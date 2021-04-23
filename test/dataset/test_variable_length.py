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
from functools import partial
from typing import Any, Dict, Iterable

import mxnet as mx
import numpy as np
import pytest

from gluonts.dataset.common import ListDataset
from gluonts.dataset.loader import (
    DataBatch,
    DataLoader,
    InferenceDataLoader,
    TrainDataLoader,
)
from gluonts.mx.batchify import _pad_arrays, batchify, stack
from gluonts.testutil.dummy_datasets import get_dataset
from gluonts.transform import (
    ContinuousTimeInstanceSplitter,
    ContinuousTimeUniformSampler,
    ContinuousTimePredictionSampler,
)


@pytest.fixture
def pp_dataset():
    return get_dataset


@pytest.fixture
def loader_factory():
    # noinspection PyTypeChecker
    def train_loader(
        dataset: ListDataset,
        prediction_interval_length: float,
        context_interval_length: float,
        is_train: bool = True,
        override_args: dict = None,
    ) -> Iterable[DataBatch]:

        if override_args is None:
            override_args = {}

        splitter = ContinuousTimeInstanceSplitter(
            future_interval_length=prediction_interval_length,
            past_interval_length=context_interval_length,
            instance_sampler=(
                ContinuousTimeUniformSampler(
                    num_instances=10,
                    min_past=context_interval_length,
                    min_future=prediction_interval_length,
                )
                if is_train
                else ContinuousTimePredictionSampler(
                    min_past=context_interval_length
                )
            ),
        )

        kwargs: Dict[str, Any] = dict(
            dataset=dataset,
            transform=splitter,
            batch_size=10,
            stack_fn=partial(
                batchify, ctx=mx.cpu(), dtype=np.float32, variable_length=True
            ),
        )
        kwargs.update(override_args)

        if is_train:
            return itertools.islice(
                TrainDataLoader(num_workers=None, **kwargs), 22
            )
        else:
            return InferenceDataLoader(**kwargs)

    return train_loader


def test_train_loader_shapes(loader_factory, pp_dataset):
    dataset = pp_dataset()
    loader = loader_factory(dataset, 1.0, 1.5)

    d = next(iter(loader))

    field_names = [
        "past_target",
        "past_valid_length",
        "future_target",
        "future_valid_length",
    ]

    assert all([key in d for key in field_names])

    assert d["past_target"].shape[2] == d["future_target"].shape[2] == 2
    assert d["past_target"].shape[0] == d["future_target"].shape[0] == 10
    assert (
        d["past_valid_length"].shape[0]
        == d["future_valid_length"].shape[0]
        == 10
    )


def test_train_loader_length(loader_factory, pp_dataset):
    dataset = pp_dataset()
    loader = loader_factory(dataset, 1.0, 1.5)

    batches = list(iter(loader))

    assert len(batches) == 22


def test_inference_loader_shapes(loader_factory, pp_dataset):
    loader = loader_factory(
        dataset=pp_dataset(),
        prediction_interval_length=1.0,
        context_interval_length=1.5,
        is_train=False,
        override_args={"batch_size": 10},
    )

    batches = list(iter(loader))

    assert len(batches) == 1

    d = batches[0]

    assert d["past_target"].shape[2] == 2
    assert d["past_target"].shape[0] == 3
    assert d["past_valid_length"].shape[0] == 3


def test_inference_loader_shapes_small_batch(loader_factory, pp_dataset):
    loader = loader_factory(
        dataset=pp_dataset(),
        prediction_interval_length=1.0,
        context_interval_length=1.5,
        is_train=False,
        override_args={"batch_size": 2},
    )

    batches = list(iter(loader))

    assert len(batches) == 2

    d = batches[0]

    assert d["past_target"].shape[2] == 2
    assert d["past_target"].shape[0] == 2
    assert d["past_valid_length"].shape[0] == 2


def test_train_loader_short_intervals(loader_factory, pp_dataset):
    loader = loader_factory(
        dataset=pp_dataset(),
        prediction_interval_length=0.001,
        context_interval_length=0.0001,
        is_train=True,
        override_args={"batch_size": 5},
    )

    batches = list(iter(loader))

    d = batches[0]

    assert d["past_target"].shape[1] == d["future_target"].shape[1] == 1
    assert d["past_target"].shape[0] == d["future_target"].shape[0] == 5


def test_inference_loader_short_intervals(loader_factory, pp_dataset):
    loader = loader_factory(
        dataset=pp_dataset(),
        prediction_interval_length=0.001,
        context_interval_length=0.0001,
        is_train=False,
        override_args={"batch_size": 5},
    )

    batches = list(iter(loader))

    d = batches[0]

    assert d["past_target"].shape[1] == 1


@pytest.mark.parametrize(
    "array_type, multi_processing",
    itertools.product(["np", "mx"], [True, False]),
)
def test_variable_length_stack(pp_dataset, array_type, multi_processing):
    arrays = [
        d["target"].T if array_type == "np" else mx.nd.array(d["target"].T)
        for d in list(iter(pp_dataset()))
    ]

    assert isinstance(multi_processing, bool)
    stacked = stack(
        arrays,
        variable_length=True,
    )

    assert stacked.shape[0] == 3
    assert stacked.shape[1] > 0
    assert stacked.shape[2] == 2


@pytest.mark.parametrize(
    "array_type, multi_processing",
    itertools.product(["np", "mx"], [True, False]),
)
def test_variable_length_stack_zerosize(
    pp_dataset, array_type, multi_processing
):
    arrays = [
        np.zeros(shape=(0, 2))
        if array_type == "np"
        else mx.nd.array(np.zeros(shape=(0, 2)))
        for _ in range(5)
    ]

    assert isinstance(multi_processing, bool)
    stacked = stack(
        arrays,
        variable_length=True,
    )

    assert stacked.shape[0] == 5
    assert stacked.shape[1] == 1
    assert stacked.shape[2] == 2


@pytest.mark.parametrize(
    "array_type, multi_processing, axis",
    itertools.product(["np", "mx"], [True, False], [0, 1]),
)
def test_pad_arrays_axis(array_type, multi_processing, axis: int):
    arrays = [
        d["target"] if array_type == "np" else mx.nd.array(d["target"])
        for d in list(iter(get_dataset()))
    ]
    if axis == 0:
        arrays = [x.T for x in arrays]

    padded_arrays = _pad_arrays(arrays, axis)

    assert all(a.shape[axis] == 8 for a in padded_arrays)
    assert all(a.shape[1 - axis] == 2 for a in padded_arrays)
