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


# Workaround needed due to a known issue with np.quantile(inp, quant) returning unsorted values.
# We fix this by ensuring that the obtained bin_centers are monotonically increasing.
# Tracked in the following issues:
# - https://github.com/numpy/numpy/issues/14685
# - https://github.com/numpy/numpy/issues/12282
def ensure_binning_monotonicity(bin_centers: np.ndarray):
    for i in range(1, len(bin_centers)):
        if bin_centers[i] < bin_centers[i - 1]:
            bin_centers[i] = bin_centers[i - 1]
    return bin_centers


def bin_edges_from_bin_centers(bin_centers: np.ndarray):
    lower_edge = -np.inf
    upper_edge = np.inf
    bin_edges = np.concatenate(
        [
            [lower_edge],
            (bin_centers[1:] + bin_centers[:-1]) / 2.0,
            [upper_edge],
        ]
    )
    return bin_edges


class Digitize(mx.operator.CustomOp):
    def forward(self, is_train, req, in_data, out_data, aux):
        data = in_data[0].asnumpy()
        bins = in_data[1].asnumpy()
        data_binned = np.digitize(data, bins=bins, right=False)
        self.assign(out_data[0], req[0], mx.nd.array(data_binned))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        return


@mx.operator.register("digitize")
class DigitizeProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(DigitizeProp, self).__init__(True)

    def list_arguments(self):
        return ["data", "bins"]

    def list_outputs(self):
        return ["output"]

    def infer_shape(self, in_shapes):
        data_shape = in_shapes[0]
        bin_shape = in_shapes[1]
        output_shape = data_shape
        return (data_shape, bin_shape), (output_shape,), ()

    def create_operator(self, ctx, in_shapes, in_dtypes):
        return Digitize()
