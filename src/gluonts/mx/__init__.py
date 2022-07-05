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

# !!! DO NOT MODIFY !!! (pkgutil-style namespace package)

# flake8: noqa

import typing
from pkgutil import extend_path

import mxnet as mx

__path__ = extend_path(__path__, __name__)  # type: ignore

# Tensor type for HybridBlocks in Gluon
Tensor = typing.Union[mx.nd.NDArray, mx.sym.Symbol]

from . import prelude as _
from .batchify import as_in_context, batchify
from .block.scaler import MeanScaler, NOPScaler
from .distribution import DistributionOutput, GaussianOutput
from .kernels import RBFKernel
from .model.estimator import GluonEstimator
from .model.predictor import (
    GluonPredictor,
    RepresentableBlockPredictor,
)
from .trainer import Trainer
from .util import copy_parameters, get_hybrid_forward_input_names

from .model.canonical import CanonicalRNNEstimator
from .model.deep_factor import DeepFactorEstimator
from .model.deepar import DeepAREstimator
from .model.deepstate import DeepStateEstimator
from .model.deepvar import DeepVAREstimator
from .model.deepvar_hierarchical import DeepVARHierarchicalEstimator
from .model.gp_forecaster import GaussianProcessEstimator
from .model.gpvar import GPVAREstimator
from .model.lstnet import LSTNetEstimator
from .model.n_beats import (
    NBEATSEstimator,
    NBEATSEnsembleEstimator,
    NBEATSEnsemblePredictor,
)

from .model.renewal import DeepRenewalProcessEstimator
from .model.san import SelfAttentionEstimator
from .model.seq2seq import (
    MQCNNEstimator,
    MQRNNEstimator,
    RNN2QRForecaster,
    Seq2SeqEstimator,
)
from .model.simple_feedforward import SimpleFeedForwardEstimator
from .model.tft import TemporalFusionTransformerEstimator
from .model.tpp import (
    PointProcessGluonPredictor,
    PointProcessSampleForecast,
    DeepTPPEstimator,
)
from .model.transformer import TransformerEstimator
from .model.wavenet import WaveNetEstimator, WaveNet, WaveNetSampler
