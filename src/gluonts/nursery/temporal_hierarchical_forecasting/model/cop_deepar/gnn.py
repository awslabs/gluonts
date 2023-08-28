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

from gluonts.mx import Tensor


class GNN(mx.gluon.HybridBlock):
    def __init__(
        self,
        units: int,
        num_layers: int,
        adj_matrix: Tensor,
        use_mlp: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.units = units
        self.num_layers = num_layers
        self.adj_matrix = adj_matrix
        self.use_mlp = use_mlp

        if self.use_mlp:
            with self.name_scope():
                self.gnn_layer = mx.gluon.nn.Dense(
                    units=self.units, flatten=False
                )

    def hybrid_forward(self, F, x, *args, **kwargs):
        # Do message passing for `num_layers` times with learnable weights.
        for _ in range(self.num_layers):
            if self.use_mlp:
                x = x + self.gnn_layer(x)
                x = F.dot(x.swapaxes(-1, -2), self.adj_matrix).swapaxes(-1, -2)
                x = F.relu(x)
            else:
                x = F.dot(x.swapaxes(-1, -2), self.adj_matrix).swapaxes(-1, -2)

        return x
