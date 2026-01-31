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

"""
MQ-DNN (Multi-Quantile Deep Neural Network) models for time series forecasting.

This package provides PyTorch implementations of:
- MQ-CNN: Multi-Quantile Convolutional Neural Network
- MQ-RNN: Multi-Quantile Recurrent Neural Network

Both models use a "forking sequence" architecture that creates multiple overlapping
training examples from a single time series to improve training efficiency.
"""

from .estimator import MQCNNEstimator, MQRNNEstimator, MQDNNEstimator
from .lightning_module import MQDNNLightningModule
from .module import (
    MQDNNModel,
    HierarchicalCausalConv1DEncoder,
    RNNEncoder,
    ForkingMLPDecoder,
    CausalConv1D,
)

__all__ = [
    "MQCNNEstimator",
    "MQRNNEstimator",
    "MQDNNEstimator",
    "MQDNNLightningModule",
    "MQDNNModel",
    "HierarchicalCausalConv1DEncoder",
    "RNNEncoder",
    "ForkingMLPDecoder",
    "CausalConv1D",
]
