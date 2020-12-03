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

import mxnet as mx
import numpy as np
import pytest

from gluonts.mx.distribution import Binned, BinnedOutput

COMMON_KWARGS = {
    "bin_log_probs": mx.nd.array([[0.1, 0.2, 0.1, 0.05, 0.2, 0.1, 0.25]])
    .log()
    .repeat(axis=0, repeats=2),
    "bin_centers": mx.nd.array([[-5, -3, -1.2, -0.5, 0, 0.1, 0.2]]).repeat(
        axis=0, repeats=2
    ),
}


@pytest.fixture
def labels():
    return mx.random.uniform(low=-6, high=1, shape=(2,))  # T, N


@pytest.mark.parametrize(
    "K,alpha", itertools.product([1000, 10000, 100000], [0.001, 0.01, 0.1])
)
def test_smooth_mask_adds_to_one(K, alpha):
    bin_log_probs = mx.nd.log_softmax(mx.nd.ones(K))
    bin_centers = mx.nd.arange(K)

    dist = Binned(
        bin_log_probs=bin_log_probs,
        bin_centers=bin_centers,
        label_smoothing=0.2,
    )

    labels = mx.random.uniform(low=0, high=K, shape=(12,)).expand_dims(-1)
    mask = dist._get_mask(labels)
    smooth_mask = dist._smooth_mask(mx.nd, mask, alpha=mx.nd.array([alpha]))

    # check smooth mask adds to one
    assert np.allclose(
        smooth_mask.asnumpy().sum(axis=-1), np.ones(12), atol=1e-6
    )


def test_get_smooth_mask_correct(labels):
    dist = Binned(**COMMON_KWARGS, label_smoothing=0.2)
    binned = Binned(**COMMON_KWARGS)

    labels = labels.expand_dims(-1)

    mask = dist._get_mask(labels)

    assert np.allclose(mask.asnumpy(), binned._get_mask(labels).asnumpy())

    smooth_mask = dist._smooth_mask(mx.nd, mask, alpha=mx.nd.array([0.2]))

    # check smooth mask adds to one
    assert np.allclose(smooth_mask.asnumpy().sum(axis=-1), np.ones(2))

    # check smooth mask peaks same
    assert np.allclose(
        np.argmax(smooth_mask.asnumpy(), axis=-1),
        np.argmax(mask.asnumpy(), axis=-1),
    )

    # check smooth mask mins correct
    assert np.allclose(
        smooth_mask.asnumpy().min(axis=-1), np.ones(2) * 0.2 / 7  # alpha / K
    )


def test_loss_correct(labels):
    smooth_alpha = Binned(**COMMON_KWARGS, label_smoothing=0.4)
    smooth_noalpha = Binned(**COMMON_KWARGS, label_smoothing=0.0)
    binned = Binned(**COMMON_KWARGS)

    assert np.allclose(
        binned.loss(labels).asnumpy(), smooth_noalpha.loss(labels).asnumpy()
    )

    assert not np.allclose(
        binned.loss(labels).asnumpy(), smooth_alpha.loss(labels).asnumpy()
    )


@pytest.mark.parametrize("hybridize", [True, False])
def test_output_sets_alpha(labels, hybridize):
    binned_output = BinnedOutput(
        bin_centers=COMMON_KWARGS["bin_centers"][0], label_smoothing=0.35
    )

    arg_proj = binned_output.get_args_proj()
    if hybridize:
        arg_proj.hybridize()
    arg_proj.initialize()

    assert (
        binned_output.distribution(
            arg_proj(mx.nd.random.uniform(2, 10))
        ).label_smoothing
        == 0.35
    )
