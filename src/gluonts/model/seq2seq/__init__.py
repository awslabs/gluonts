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

import warnings

warnings.warn(
    "The seq2seq model in gulonts.model is deprecated and will be moved to "
    "'gluonts.mx.model'. Try to use 'from gluonts.mx import "
    "MQCNNEstimator, MQRNNEstimator, RNN2QRForecaster, Seq2SeqEstimator'.",
    FutureWarning,
)

from gluonts.mx.model.seq2seq._mq_dnn_estimator import (
    MQCNNEstimator,
    MQRNNEstimator,
)
from gluonts.mx.model.seq2seq._seq2seq_estimator import (
    RNN2QRForecaster,
    Seq2SeqEstimator,
)

__all__ = [
    "MQCNNEstimator",
    "MQRNNEstimator",
    "RNN2QRForecaster",
    "Seq2SeqEstimator",
]
