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
import mxnet as mx
import numpy as np
from mxnet import nd

# First-party imports
from gluonts.block.cnn import CausalConv1D


def compute_causalconv1d(
    x: np.ndarray, kernels: np.ndarray, dilation: int
) -> np.ndarray:
    """
    Naive way to compute the 1-d causal convolution

    Parameters:
    x: np.array
      input array
    kernels: np.array
      array of weights, [w_1, ..., w_n] where n is the kernel_size
    dilation: int
      dilation rate d > 0, when d = 1, it reduces to the regular convolution
    Returns:
      Causal 1d convolution between x and kernels with the given dilation rate
    """

    conv_x = np.zeros_like(x)
    # compute in a naive way.
    for (t, xt) in enumerate(x):
        dial_offset = 0
        for i in reversed(range(len(kernels))):
            xt_lag = x[t - dial_offset] if t - dial_offset >= 0 else 0.0
            dial_offset += dilation
            conv_x[t] += kernels[i] * xt_lag

    return conv_x


def test_causal_conv_1d() -> None:
    """
    Here we test whether the causal conv1d matches the naive computation.
    """
    x = nd.random.normal(0, 1, shape=(1, 1, 10))

    conv1d = CausalConv1D(
        channels=1, kernel_size=3, dilation=3, activation=None
    )
    conv1d.collect_params().initialize(
        mx.init.One(), ctx=mx.cpu(), force_reinit=True
    )

    y1 = conv1d(x).reshape(shape=(-1,)).asnumpy()
    y2 = compute_causalconv1d(
        x=x.reshape(shape=(-1,)).asnumpy(),
        kernels=np.asarray([1.0] * 3),
        dilation=3,
    )

    assert (
        np.max(np.abs(y1 - y2)) < 1e-5
    ), "1d Causal Convolution calculation incorrect!"
