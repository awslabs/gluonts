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

# Standard library imports
from typing import Optional

# Third-party imports
import numpy as np
from mxnet.gluon import HybridBlock, loss

# First-party imports
from gluonts.core.component import validated, DType
from gluonts.model.estimator import GluonEstimator
from gluonts.model.lstnet._network import LSTNetTrain, LSTNetPredict
from gluonts.model.predictor import Predictor, RepresentableBlockPredictor
from gluonts.trainer import Trainer
from gluonts.dataset.field_names import FieldName
from gluonts.support.util import copy_parameters
from gluonts.transform import (
    Chain,
    ExpectedNumInstanceSampler,
    InstanceSplitter,
    Transformation,
    AsNumpyArray,
    AddObservedValuesIndicator,
)


class LSTNetEstimator(GluonEstimator):
    """
    Constructs an LSTNet estimator for multivariate time-series data.

    The model has been described in this paper:
    https://arxiv.org/abs/1703.07015

    Note that this implementation will change over time as we further work on
    this method.

    Parameters
    ----------
    freq
        Frequency of the data to train and predict
    prediction_length
        Length of the prediction p where given `(y_1, ..., y_t)` the model
        predicts `(y_{t+l+1}, ..., y_{t+l+p})`, where l is `lead_time`
    context_length
        The maximum number of steps to unroll the RNN for computing the
        predictions
        (Note that it is constraints by the Conv2D output size)
    num_series
        Number of time-series (covariates)
    skip_size
        Skip size for the skip RNN layer
    ar_window
        Auto-regressive window size for the linear part
    channels
        Number of channels for first layer Conv2D
    lead_time
        Lead time (default: 0)
    kernel_size
        Kernel size for first layer Conv2D (default: 6)
    trainer
        Trainer object to be used (default: Trainer())
    dropout_rate
        Dropout regularization parameter (default: 0.2)
    output_activation
        The last activation to be used for output.
        Accepts either `None` (default no activation), `sigmoid` or `tanh`
    rnn_cell_type
        Type of the RNN cell. Either `lstm` or `gru` (default: `gru`)
    rnn_num_layers
        Number of RNN layers to be used
    rnn_num_cells
        Number of RNN cells for each layer (default: 100)
    skip_rnn_cell_type
        Type of the RNN cell for the skip layer. Either `lstm` or `gru` (
        default: `gru`)
    skip_rnn_num_layers
        Number of RNN layers to be used for skip part
    skip_rnn_num_cells
        Number of RNN cells for each layer for skip part (default: 10)
    scaling
        Whether to automatically scale the target values (default: True)
    dtype
        Data type (default: np.float32)
    """

    @validated()
    def __init__(
        self,
        freq: str,
        prediction_length: int,
        context_length: int,
        num_series: int,
        skip_size: int,
        ar_window: int,
        channels: int,
        lead_time: int = 0,
        kernel_size: int = 6,
        trainer: Trainer = Trainer(),
        dropout_rate: Optional[float] = 0.2,
        output_activation: Optional[str] = None,
        rnn_cell_type: str = "gru",
        rnn_num_cells: int = 100,
        rnn_num_layers: int = 3,
        skip_rnn_cell_type: str = "gru",
        skip_rnn_num_layers: int = 1,
        skip_rnn_num_cells: int = 10,
        scaling: bool = True,
        dtype: DType = np.float32,
    ) -> None:
        super().__init__(trainer=trainer, lead_time=lead_time, dtype=dtype)
        self.freq = freq
        self.num_series = num_series
        self.skip_size = skip_size
        self.ar_window = ar_window
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.channels = channels
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        self.output_activation = output_activation
        self.rnn_cell_type = rnn_cell_type
        self.rnn_num_layers = rnn_num_layers
        self.rnn_num_cells = rnn_num_cells
        self.skip_rnn_cell_type = skip_rnn_cell_type
        self.skip_rnn_num_layers = skip_rnn_num_layers
        self.skip_rnn_num_cells = skip_rnn_num_cells
        self.scaling = scaling
        self.dtype = dtype

    def create_transformation(self) -> Transformation:
        return Chain(
            trans=[
                AsNumpyArray(
                    field=FieldName.TARGET, expected_ndim=2, dtype=self.dtype
                ),
                AddObservedValuesIndicator(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.OBSERVED_VALUES,
                    dtype=self.dtype,
                ),
                InstanceSplitter(
                    target_field=FieldName.TARGET,
                    is_pad_field=FieldName.IS_PAD,
                    start_field=FieldName.START,
                    forecast_start_field=FieldName.FORECAST_START,
                    train_sampler=ExpectedNumInstanceSampler(num_instances=1),
                    time_series_fields=[FieldName.OBSERVED_VALUES],
                    past_length=self.context_length,
                    future_length=self.prediction_length,
                    lead_time=self.lead_time,
                    output_NTC=False,  # output NCT for first layer conv2d
                ),
            ]
        )

    def create_training_network(self) -> HybridBlock:
        return LSTNetTrain(
            num_series=self.num_series,
            channels=self.channels,
            kernel_size=self.kernel_size,
            rnn_cell_type=self.rnn_cell_type,
            rnn_num_layers=self.rnn_num_layers,
            rnn_num_cells=self.rnn_num_cells,
            skip_rnn_cell_type=self.skip_rnn_cell_type,
            skip_rnn_num_layers=self.skip_rnn_num_layers,
            skip_rnn_num_cells=self.skip_rnn_num_cells,
            skip_size=self.skip_size,
            ar_window=self.ar_window,
            context_length=self.context_length,
            lead_time=self.lead_time,
            prediction_length=self.prediction_length,
            dropout_rate=self.dropout_rate,
            output_activation=self.output_activation,
            scaling=self.scaling,
            dtype=self.dtype,
        )

    def create_predictor(
        self, transformation: Transformation, trained_network: HybridBlock
    ) -> Predictor:
        prediction_network = LSTNetPredict(
            num_series=self.num_series,
            channels=self.channels,
            kernel_size=self.kernel_size,
            rnn_cell_type=self.rnn_cell_type,
            rnn_num_layers=self.rnn_num_layers,
            rnn_num_cells=self.rnn_num_cells,
            skip_rnn_cell_type=self.skip_rnn_cell_type,
            skip_rnn_num_layers=self.skip_rnn_num_layers,
            skip_rnn_num_cells=self.skip_rnn_num_cells,
            skip_size=self.skip_size,
            ar_window=self.ar_window,
            context_length=self.context_length,
            lead_time=self.lead_time,
            prediction_length=self.prediction_length,
            dropout_rate=self.dropout_rate,
            output_activation=self.output_activation,
            scaling=self.scaling,
            dtype=self.dtype,
        )

        copy_parameters(trained_network, prediction_network)

        return RepresentableBlockPredictor(
            input_transform=transformation,
            prediction_net=prediction_network,
            batch_size=self.trainer.batch_size,
            freq=self.freq,
            prediction_length=self.prediction_length,
            lead_time=self.lead_time,
            ctx=self.trainer.ctx,
            dtype=self.dtype,
        )
