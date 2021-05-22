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

from mxnet.gluon import HybridBlock, nn

from gluonts.core.component import validated
from gluonts.mx.block.rnn import RNN


class RNNModel(HybridBlock):
    @validated()
    def __init__(
        self,
        mode,
        num_hidden,
        num_layers,
        num_output,
        bidirectional=False,
        **kwargs,
    ):
        super(RNNModel, self).__init__(**kwargs)
        self.num_output = num_output

        with self.name_scope():
            self.rnn = RNN(
                mode=mode,
                num_hidden=num_hidden,
                num_layers=num_layers,
                bidirectional=bidirectional,
            )

            self.decoder = nn.Dense(
                num_output, in_units=num_hidden, flatten=False
            )

    def hybrid_forward(self, F, inputs):
        return self.decoder(self.rnn(inputs))
