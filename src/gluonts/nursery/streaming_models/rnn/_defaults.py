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

from typing import Dict, List, Optional, Tuple, Union

import mxnet as mx
import numpy as np

from gluonts.mx.distribution import BinnedOutput, DistributionOutput
from gluonts.mx.trainer.model_iteration_averaging import Alpha_Suffix
from gluonts.transform import DummyValueImputation, MissingValueImputation

bin_centers = mx.nd.array(
    np.concatenate([np.linspace(0, 2, 1001), np.linspace(2.02, 10, 501)])
)


class RnnDefaults:
    FREQ: str = "1min"
    LEAD_TIME: int = 5 * 60
    TRAIN_WINDOW_LENGTH: int = 2000
    SKIP_INITIAL_WINDOW_PCT: float = 0.1
    HIDDEN_SIZE: int = 40
    NUM_LAYERS: int = 2
    DISTR_OUTPUT: DistributionOutput = BinnedOutput(bin_centers=bin_centers)
    USE_FEAT_STATIC_CAT: bool = False
    CARDINALITY: Optional[List[int]] = None
    EMBEDDING_DIMENSION: Optional[List[int]] = None
    LAGS: Optional[List[Tuple[str, List[int], str]]] = None
    DROPOUT_TYPE: str = "normal"
    DROPOUT_RATE: float = 0.1
    BATCH_SIZE: int = 32
    TRAINER_KWARGS: Dict = dict(
        epochs=200,
        num_batches_per_epoch=300,
        learning_rate=2e-3,
        avg_strategy=Alpha_Suffix(200, alpha=0.4),
    )
    ALPHA: float = 0.0
    BETA: float = 0.0
    IMPUTATION: Optional[
        Union[str, MissingValueImputation]
    ] = DummyValueImputation()
    RANSAC_THRESH: Optional[List[float]] = None
    HYBRIDIZE_PRED_NET: bool = True
