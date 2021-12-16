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

from gluonts.dataset.loader import InferenceDataLoader
from gluonts.torch.batchify import batchify
from gluonts.torch.util import get_forward_input_names

from ._network import DeepNPTSNetworkDiscrete


def forecast_probabilities(predictor, dataset):
    inference_data_loader = InferenceDataLoader(
        dataset,
        transform=predictor.input_transform,
        batch_size=predictor.batch_size,
        stack_fn=lambda data: batchify(data, predictor.device),
    )
    net = predictor.prediction_net.net
    assert isinstance(net, DeepNPTSNetworkDiscrete)
    input_names = get_forward_input_names(DeepNPTSNetworkDiscrete)
    for batch in inference_data_loader:
        x = {k: batch[k] for k in input_names}
        distr = predictor.prediction_net.net(**x)
        for k in range(distr.probs.shape[0]):
            yield distr.probs[k].detach().numpy()
