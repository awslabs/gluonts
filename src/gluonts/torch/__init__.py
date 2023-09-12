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


import torch

from gluonts.meta.export import re_export

__all__ = re_export(
    __name__,
    ".model.estimator.PyTorchLightningEstimator",
    ".model.predictor.PyTorchPredictor",
    ".model.deep_npts.DeepNPTSEstimator",
    ".model.deepar.DeepAREstimator",
    ".model.simple_feedforward.SimpleFeedForwardEstimator",
    ".model.tft.TemporalFusionTransformerEstimator",
    ".model.wavenet.WaveNetEstimator",
    ".model.d_linear.DLinearEstimator",
    ".model.patch_tst.PatchTSTEstimator",
    ".model.lag_tst.LagTSTEstimator",
)


from . import prelude as _  # noqa


if torch.cuda.is_available():
    torch.multiprocessing.set_start_method("spawn", force=True)
