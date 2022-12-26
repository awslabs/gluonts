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

from collections import OrderedDict
import torch.nn as nn
import torch


def load_weights(model: nn.Module, path_to_weights: str) -> nn.Module:
    ckpt = torch.load(
        path_to_weights, map_location=lambda storage, loc: storage
    )

    # this seems to be necessary
    for key in ckpt["state_dict"].copy():
        if ("loss" in key) or ("crps" in key) or ("quantile_width" in key):
            ckpt["state_dict"].pop(key)

    new_state_dict = OrderedDict()
    for k, v in ckpt["state_dict"].items():
        name = k.replace("model.", "")
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    return model
