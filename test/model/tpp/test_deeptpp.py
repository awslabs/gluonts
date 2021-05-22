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

import mxnet as mx
import numpy as np

import pytest
from mxnet import nd

from gluonts.model.tpp.deeptpp._network import (
    DeepTPPPredictionNetwork,
    DeepTPPTrainingNetwork,
)
from gluonts.model.tpp.distribution import WeibullOutput


def _allclose(a: nd.NDArray, b: nd.NDArray):
    return np.allclose(a.asnumpy(), b.asnumpy(), atol=1e-6)


TEST_CASES = [  # ia_times, marks, valid_length, num_marks, loglike
    (
        nd.array([[0.1, 0.2, 0.1, 0.12], [0.3, 0.15, 0.1, 0.12]]),
        nd.array([[1, 2, 0, 2], [0, 0, 1, 2]]),
        nd.array([3, 4]),
        3,
        nd.array([5.0570173, 6.6502876]),
    ),
    (
        nd.array([[0.1, 0.2, 0.1, 0.12], [0.3, 0.15, 0.1, 0.12]]),
        nd.array([[1, 2, 0, 2], [0, 0, 1, 2]]),
        nd.array([0, 4]),
        3,
        nd.array([0.9517526, 6.6502876]),
    ),
    (
        nd.array([[0.1, 0.2, 0.1, 0.12], [0.3, 0.15, 0.1, 0.12]]),
        nd.array([[1, 2, 0, 2], [0, 0, 1, 2]]),
        nd.array([0, 0]),
        3,
        nd.array([0.9517526, 0.9725293]),
    ),
    (
        nd.array([[0.1, 0.2, 0.1, 0.12], [0.3, 0.15, 0.1, 0.12]]),
        nd.array([[0, 0, 0, 0], [0, 0, 0, 0]]),
        nd.array([3, 4]),
        1,
        nd.array([1.7608211, 2.2495322]),
    ),
]


@pytest.mark.parametrize(
    "ia_times, marks, valid_length, num_marks, loglike", TEST_CASES
)
def test_log_likelihood(ia_times, marks, valid_length, num_marks, loglike):
    mx.rnd.seed(seed_state=1234)

    model = DeepTPPTrainingNetwork(
        num_marks=num_marks,
        interval_length=2,
        time_distr_output=WeibullOutput(),
    )
    model.initialize()

    loglike_pred = model(nd.stack(ia_times, marks, axis=-1), valid_length)

    assert loglike_pred.shape == (ia_times.shape[0],)
    assert _allclose(loglike, loglike_pred)


def test_trainining_network_disallows_hybrid():
    mx.rnd.seed(seed_state=1234)

    with pytest.raises(NotImplementedError):
        smodel = DeepTPPTrainingNetwork(num_marks=3, interval_length=2)
        smodel.hybridize()


def test_prediction_network_disallows_hybrid():
    mx.rnd.seed(seed_state=1234)

    with pytest.raises(NotImplementedError):
        smodel = DeepTPPPredictionNetwork(
            num_marks=3, interval_length=2, prediction_interval_length=3
        )
        smodel.hybridize()


def test_prediction_network_output():
    mx.rnd.seed(seed_state=1234)
    model = DeepTPPPredictionNetwork(
        num_marks=5,
        time_distr_output=WeibullOutput(),
        interval_length=1.0,
        prediction_interval_length=10.0,
    )
    model.initialize()
    past_ia_times = nd.array([[0.1, 0.2, 0.1, 0.12], [0.3, 0.15, 0.1, 0.12]])
    past_marks = nd.array([[1, 2, 0, 2], [0, 0, 1, 2]])
    past_valid_length = nd.array([3, 4])
    past_target = nd.stack(past_ia_times, past_marks, axis=-1)

    pred_target, pred_valid_length = model(past_target, past_valid_length)

    # pred_target must have shape
    # (num_parallel_samples, batch_size, max_sequence_length, 2)
    assert pred_target.ndim == 4
    assert pred_target.shape[0] == model.num_parallel_samples
    assert pred_target.shape[1] == past_ia_times.shape[0]
    assert pred_target.shape[3] == 2  # TPP prediction contains ia_time & mark
    # pred_valid_length must have shape (num_parallel_samples, batch_size)
    assert pred_valid_length.ndim == 2
    assert pred_valid_length.shape[0] == model.num_parallel_samples
    assert pred_valid_length.shape[1] == past_ia_times.shape[0]

    pred_ia_times = pred_target[..., 0].asnumpy()
    pred_marks = pred_target[..., 1].asnumpy()

    assert pred_marks.min() >= 0
    assert pred_marks.max() < model.num_marks
    assert (pred_ia_times >= 0).all()
    # ia_times are set to zero above valid_length (see DeepTPPPredictionNetwork)
    assert (pred_ia_times.sum(-1) < model.prediction_interval_length).all()
