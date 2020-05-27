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
import pytest
import mxnet as mx
import numpy as np
from mxnet import nd

# First-party imports
from gluonts.model.tpp.rmtpp._network import RMTPPTrainingNetwork


def _allclose(a: nd.NDArray, b: nd.NDArray):
    return np.allclose(a.asnumpy(), b.asnumpy(), atol=1e-6)


TEST_CASES = [  # ia_times, marks, valid_length, expected_ll
    (
        nd.array([[0.1, 0.2, 0.1, 0.12], [0.3, 0.15, 0.1, 0.12]]),
        nd.array([[1, 2, 0, 2], [0, 0, 1, 2]]),
        nd.array([3, 4]),
        False,
    ),
    (
        nd.array([[0.1, 0.2, 0.1, 0.12], [0.3, 0.15, 0.1, 0.12]]),
        nd.array([[1, 2, 0, 2], [0, 0, 1, 2]]),
        nd.array([0, 4]),
        False,
    ),
    (
        nd.array([[0.1, 0.2, 0.1, 0.12], [0.3, 0.15, 0.1, 0.12]]),
        nd.array([[1, 2, 0, 2], [0, 0, 1, 2]]),
        nd.array([0, 0]),
        True,
    ),
]


@pytest.mark.parametrize("ia_times, marks, valid_length, expected_ll", TEST_CASES)
def test_log_likelihood(ia_times, marks, valid_length, expected_ll):
    mx.rnd.seed(seed_state=1234)

    smodel = RMTPPTrainingNetwork(
        num_marks=3, interval_length=2
    )

    smodel.collect_params().initialize()

    ll = smodel(
        nd.stack(ia_times, marks, axis=-1),
        valid_length,
    )

    if expected_ll:
        beta = -nd.Activation(smodel.decay_bias.data(), "softrelu")
        ll_pred = -(
            nd.reciprocal(beta) * nd.expm1(nd.broadcast_mul(nd.array([2, 2]), beta))
        ).broadcast_like(ll)
        assert _allclose(ll, ll_pred)


def test_rmtpp_disallows_hybrid():
    mx.rnd.seed(seed_state=1234)

    with pytest.raises(NotImplementedError):
        smodel = RMTPPTrainingNetwork(
            num_marks=3,
            interval_length=2,
        )
        smodel.hybridize()


#
# @pytest.mark.parametrize("hybridize", [True, False])
# def test_log_likelihood_max_time(hybridize):
#     mx.rnd.seed(seed_state=1234)
#
#     smodel = RMTPPBlock(num_marks=3, sequence_length=2)
#     if hybridize:
#         smodel.hybridize()
#
#     smodel.collect_params().initialize()
#
#     lags = nd.array([[0.1, 0.2, 0.1, 0.12], [0.3, 0.15, 0.1, 0.12]])
#     marks = nd.array([[1, 2, 0, 2], [0, 0, 1, 2]])
#
#     valid_length = nd.ones(shape=(4,)) * 2
#     max_time = nd.ones(shape=(4,)) * 5
#
#     assert _allclose(
#         smodel(lags, marks, valid_length, max_time),
#         nd.array([-4.2177677, -4.146564, -3.911864, -3.9754848]),
#     )
#
#
# @pytest.mark.parametrize("hybridize", [True, False])
# def test_log_likelihood_valid_length(hybridize):
#     mx.rnd.seed(seed_state=1234)
#
#     smodel = RMTPPBlock(num_marks=3, sequence_length=2)
#     if hybridize:
#         smodel.hybridize()
#
#     smodel.collect_params().initialize()
#
#     lags = nd.array([[0.1, 0.2, 0.1, 0.12], [0.3, 0.15, 0.1, 0.12]])
#     marks = nd.array([[1, 2, 0, 2], [0, 0, 1, 2]])
#
#     valid_length = nd.array([1, 2, 1, 1])
#     max_time = nd.ones(shape=(4,)) * 5
#
#     assert _allclose(
#         smodel(lags, marks, valid_length, max_time),
#         nd.array([-2.6500425, -4.146564, -2.6500664, -2.6819286]),
#     )
#
#
# @pytest.mark.parametrize("hybridize", [True, False])
# def test_sampler_shapes_correct(hybridize):
#     mx.rnd.seed(seed_state=1234)
#
#     smodel = RMTPPBlock(num_marks=3, sequence_length=2)
#     if hybridize:
#         smodel.hybridize()
#
#     smodel.collect_params().initialize()
#
#     lags = nd.array([[0.1, 0.2, 0.1, 0.12], [0.3, 0.15, 0.1, 0.12]])
#     marks = nd.array([[1, 2, 0, 2], [0, 0, 1, 2]])
#     valid_length = nd.array([1, 2, 1, 1])
#     max_time = nd.ones(shape=(4,)) * 5
#
#     smodel(lags, marks, valid_length, max_time)
#
#     sampler = RMTPPSampler(smodel)
#
#     ia_times, marks, valid_length_samp = sampler.ogata_sample(
#         5., batch_size=12
#     )
#
#     assert marks.asnumpy().max() < smodel.num_marks
#     assert marks.shape[1] == 12
#     assert ia_times.shape[1] == 12
#     assert valid_length_samp.shape[0] == 12
