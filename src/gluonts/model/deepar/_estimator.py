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
import numpy as np
from typing import List, Optional

# Third-party imports
from mxnet.gluon import HybridBlock

# First-party imports
from gluonts.core.component import DType, validated
from gluonts.dataset.field_names import FieldName
from gluonts.distribution import DistributionOutput, StudentTOutput
from gluonts.model.estimator import GluonEstimator
from gluonts.model.predictor import Predictor, RepresentableBlockPredictor
from gluonts.support.util import copy_parameters
from gluonts.time_feature import (
    TimeFeature,
    time_features_from_frequency_str,
    get_lags_for_frequency,
)
from gluonts.trainer import Trainer
from gluonts.transform import (
    AddAgeFeature,
    AddObservedValuesIndicator,
    AddTimeFeatures,
    AsNumpyArray,
    Chain,
    ExpectedNumInstanceSampler,
    InstanceSplitter,
    RemoveFields,
    SetField,
    Transformation,
    VstackFeatures,
)

# Relative imports
from ._network import DeepARPredictionNetwork, DeepARTrainingNetwork


class DeepAREstimator(GluonEstimator):
    """
    Construct a DeepAR estimator.

    This implements an RNN-based model, close to the one described in
    [SFG17]_.

    *Note:* the code of this model is unrelated to the implementation behind
    `SageMaker's DeepAR Forecasting Algorithm
    <https://docs.aws.amazon.com/sagemaker/latest/dg/deepar.html>`_.

    Parameters
    ----------
    freq
        Frequency of the data to train on and predict
    prediction_length
        Length of the prediction horizon
    trainer
        Trainer object to be used (default: Trainer())
    context_length
        Number of steps to unroll the RNN for before computing predictions
        (default: None, in which case context_length = prediction_length)
    num_layers
        Number of RNN layers (default: 2)
    num_cells
        Number of RNN cells for each layer (default: 40)
    cell_type
        Type of recurrent cells to use (available: 'lstm' or 'gru';
        default: 'lstm')
    dropout_rate
        Dropout regularization parameter (default: 0.1)
    use_feat_dynamic_real
        Whether to use the ``feat_dynamic_real`` field from the data
        (default: False)
    use_feat_static_cat
        Whether to use the ``feat_static_cat`` field from the data
        (default: False)
    use_feat_static_real
        Whether to use the ``feat_static_real`` field from the data
        (default: False)
    cardinality
        Number of values of each categorical feature.
        This must be set if ``use_feat_static_cat == True`` (default: None)
    embedding_dimension
        Dimension of the embeddings for categorical features
        (default: [min(50, (cat+1)//2) for cat in cardinality])
    distr_output
        Distribution to use to evaluate observations and sample predictions
        (default: StudentTOutput())
    scaling
        Whether to automatically scale the target values (default: true)
    lags_seq
        Indices of the lagged target values to use as inputs of the RNN
        (default: None, in which case these are automatically determined
        based on freq)
    time_features
        Time features to use as inputs of the RNN (default: None, in which
        case these are automatically determined based on freq)
    num_parallel_samples
        Number of evaluation samples per time series to increase parallelism during inference.
        This is a model optimization that does not affect the accuracy (default: 100)
    """

    @validated()
    def __init__(
        self,
        freq: str,
        prediction_length: int,
        trainer: Trainer = Trainer(),
        context_length: Optional[int] = None,
        num_layers: int = 2,
        num_cells: int = 40,
        cell_type: str = "lstm",
        dropout_rate: float = 0.1,
        use_feat_dynamic_real: bool = False,
        use_feat_static_cat: bool = False,
        use_feat_static_real: bool = False,
        cardinality: Optional[List[int]] = None,
        embedding_dimension: Optional[List[int]] = None,
        distr_output: DistributionOutput = StudentTOutput(),
        scaling: bool = True,
        lags_seq: Optional[List[int]] = None,
        time_features: Optional[List[TimeFeature]] = None,
        num_parallel_samples: int = 100,
        dtype: DType = np.float32,
    ) -> None:
        super().__init__(trainer=trainer, dtype=dtype)

        assert (
            prediction_length > 0
        ), "The value of `prediction_length` should be > 0"
        assert (
            context_length is None or context_length > 0
        ), "The value of `context_length` should be > 0"
        assert num_layers > 0, "The value of `num_layers` should be > 0"
        assert num_cells > 0, "The value of `num_cells` should be > 0"
        assert dropout_rate >= 0, "The value of `dropout_rate` should be >= 0"
        assert (cardinality is not None and use_feat_static_cat) or (
            cardinality is None and not use_feat_static_cat
        ), "You should set `cardinality` if and only if `use_feat_static_cat=True`"
        assert cardinality is None or all(
            [c > 0 for c in cardinality]
        ), "Elements of `cardinality` should be > 0"
        assert embedding_dimension is None or all(
            [e > 0 for e in embedding_dimension]
        ), "Elements of `embedding_dimension` should be > 0"
        assert (
            num_parallel_samples > 0
        ), "The value of `num_parallel_samples` should be > 0"

        self.freq = freq
        self.context_length = (
            context_length if context_length is not None else prediction_length
        )
        self.prediction_length = prediction_length
        self.distr_output = distr_output
        self.distr_output.dtype = dtype
        self.num_layers = num_layers
        self.num_cells = num_cells
        self.cell_type = cell_type
        self.dropout_rate = dropout_rate
        self.use_feat_dynamic_real = use_feat_dynamic_real
        self.use_feat_static_cat = use_feat_static_cat
        self.use_feat_static_real = use_feat_static_real
        self.cardinality = (
            cardinality if cardinality and use_feat_static_cat else [1]
        )
        self.embedding_dimension = (
            embedding_dimension
            if embedding_dimension is not None
            else [min(50, (cat + 1) // 2) for cat in self.cardinality]
        )
        self.scaling = scaling
        self.lags_seq = (
            lags_seq
            if lags_seq is not None
            else get_lags_for_frequency(freq_str=freq)
        )
        self.time_features = (
            time_features
            if time_features is not None
            else time_features_from_frequency_str(self.freq)
        )

        self.history_length = self.context_length + max(self.lags_seq)

        self.num_parallel_samples = num_parallel_samples

    def create_transformation(self) -> Transformation:
        remove_field_names = [FieldName.FEAT_DYNAMIC_CAT]
        if not self.use_feat_static_real:
            remove_field_names.append(FieldName.FEAT_STATIC_REAL)
        if not self.use_feat_dynamic_real:
            remove_field_names.append(FieldName.FEAT_DYNAMIC_REAL)

        return Chain(
            [RemoveFields(field_names=remove_field_names)]
            + (
                [SetField(output_field=FieldName.FEAT_STATIC_CAT, value=[0.0])]
                if not self.use_feat_static_cat
                else []
            )
            + (
                [
                    SetField(
                        output_field=FieldName.FEAT_STATIC_REAL, value=[0.0]
                    )
                ]
                if not self.use_feat_static_real
                else []
            )
            + [
                AsNumpyArray(
                    field=FieldName.FEAT_STATIC_CAT,
                    expected_ndim=1,
                    dtype=self.dtype,
                ),
                AsNumpyArray(
                    field=FieldName.FEAT_STATIC_REAL,
                    expected_ndim=1,
                    dtype=self.dtype,
                ),
                AsNumpyArray(
                    field=FieldName.TARGET,
                    # in the following line, we add 1 for the time dimension
                    expected_ndim=1 + len(self.distr_output.event_shape),
                    dtype=self.dtype,
                ),
                AddObservedValuesIndicator(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.OBSERVED_VALUES,
                    dummy_value=self.distr_output.value_in_support,
                    dtype=self.dtype,
                ),
                AddTimeFeatures(
                    start_field=FieldName.START,
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_TIME,
                    time_features=self.time_features,
                    pred_length=self.prediction_length,
                ),
                AddAgeFeature(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_AGE,
                    pred_length=self.prediction_length,
                    log_scale=True,
                    dtype=self.dtype,
                ),
                VstackFeatures(
                    output_field=FieldName.FEAT_TIME,
                    input_fields=[FieldName.FEAT_TIME, FieldName.FEAT_AGE]
                    + (
                        [FieldName.FEAT_DYNAMIC_REAL]
                        if self.use_feat_dynamic_real
                        else []
                    ),
                ),
                InstanceSplitter(
                    target_field=FieldName.TARGET,
                    is_pad_field=FieldName.IS_PAD,
                    start_field=FieldName.START,
                    forecast_start_field=FieldName.FORECAST_START,
                    train_sampler=ExpectedNumInstanceSampler(num_instances=1),
                    past_length=self.history_length,
                    future_length=self.prediction_length,
                    time_series_fields=[
                        FieldName.FEAT_TIME,
                        FieldName.OBSERVED_VALUES,
                    ],
                    dummy_value=self.distr_output.value_in_support,
                ),
            ]
        )

    def create_training_network(self) -> DeepARTrainingNetwork:
        return DeepARTrainingNetwork(
            num_layers=self.num_layers,
            num_cells=self.num_cells,
            cell_type=self.cell_type,
            history_length=self.history_length,
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            distr_output=self.distr_output,
            dropout_rate=self.dropout_rate,
            cardinality=self.cardinality,
            embedding_dimension=self.embedding_dimension,
            lags_seq=self.lags_seq,
            scaling=self.scaling,
            dtype=self.dtype,
        )

    def create_predictor(
        self, transformation: Transformation, trained_network: HybridBlock
    ) -> Predictor:
        prediction_network = DeepARPredictionNetwork(
            num_parallel_samples=self.num_parallel_samples,
            num_layers=self.num_layers,
            num_cells=self.num_cells,
            cell_type=self.cell_type,
            history_length=self.history_length,
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            distr_output=self.distr_output,
            dropout_rate=self.dropout_rate,
            cardinality=self.cardinality,
            embedding_dimension=self.embedding_dimension,
            lags_seq=self.lags_seq,
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
            ctx=self.trainer.ctx,
            dtype=self.dtype,
        )
