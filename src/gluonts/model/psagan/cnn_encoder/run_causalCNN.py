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

import logging

from gluonts.dataset.repository.datasets import get_dataset
from gluonts.model.psagan.cnn_encoder._estimator import (
    CausalCNNEncoderEstimator,
)
from gluonts.model.psagan.cnn_encoder._trainer import Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S]",
)
logger = logging.getLogger(__name__)

dataset = get_dataset("m4_hourly")

estimator = CausalCNNEncoderEstimator(
    trainer=Trainer(num_epochs=10, lr=0.001, save_display_frq=1, device="cpu"),
    freq=dataset.metadata.freq,
    batch_size=64,
    nb_features=1,
    nb_channels=40,
    depth=3,
    reduced_size=160,
    size_embedding=80,
    kernel_size=3,
    subseries_length=20,
    context_length=40,
    max_len=250,
    nb_negative_samples=10,
    use_feat_dynamic_real=True,
    use_feat_static_cat=True,
    use_feat_static_real=True,
    device="cpu",
)


estimator.train(dataset.train)
