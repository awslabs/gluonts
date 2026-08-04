# Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import torch

from pytorch_lightning.metrics import Metric
from pytorch_lightning.utilities import rank_zero_warn


class CachePredictions(Metric):
    """Compute a number of metrics for  over all batches"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.add_state("preds", default=[], dist_reduce_fx=None)
        self.add_state("target", default=[], dist_reduce_fx=None)

        rank_zero_warn(
            "Metric `CachePredictions` will save all targets and predictions in buffer."
            " For large datasets this may lead to large memory footprint."
        )

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # preds, target = _input_format_classification(preds, target)
        assert preds.shape == target.shape

        self.preds.append(preds)
        self.target.append(target)

    def compute(self):
        return self.preds, self.target
