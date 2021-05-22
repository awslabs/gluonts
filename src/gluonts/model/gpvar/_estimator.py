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

from functools import partial
from typing import List, Optional

from mxnet.gluon import HybridBlock

from gluonts.core.component import validated
from gluonts.dataset.common import Dataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.loader import (
    DataLoader,
    TrainDataLoader,
    ValidationDataLoader,
)
from gluonts.model.deepvar._estimator import (
    get_lags_for_frequency,
    time_features_from_frequency_str,
)
from gluonts.mx.batchify import batchify, as_in_context
from gluonts.mx.model.estimator import GluonEstimator
from gluonts.model.predictor import Predictor
from gluonts.mx.distribution import DistributionOutput
from gluonts.mx.distribution.lowrank_gp import LowrankGPOutput
from gluonts.mx.model.predictor import RepresentableBlockPredictor
from gluonts.mx.trainer import Trainer
from gluonts.mx.util import copy_parameters, get_hybrid_forward_input_names
from gluonts.time_feature import TimeFeature
from gluonts.transform import (
    AddObservedValuesIndicator,
    AddTimeFeatures,
    AsNumpyArray,
    CDFtoGaussianTransform,
    Chain,
    ExpandDimArray,
    ExpectedNumInstanceSampler,
    InstanceSplitter,
    RenameFields,
    SampleTargetDim,
    SetFieldIfNotPresent,
    SelectFields,
    TargetDimIndicator,
    Transformation,
    VstackFeatures,
    cdf_to_gaussian_forward_transform,
    InstanceSampler,
    ValidationSplitSampler,
    TestSplitSampler,
)

from ._network import GPVARPredictionNetwork, GPVARTrainingNetwork


class GPVAREstimator(GluonEstimator):

    """
    Constructs a GPVAR estimator.

    These models have been described as GP-Copula in this paper:
    https://arxiv.org/abs/1910.03002

    Note that this implementation will change over time and we further work on
    this method. To replicate the results of the paper, please refer to our
    (frozen) implementation here:
    https://github.com/mbohlkeschneider/gluon-ts/tree/mv_release


    Parameters
    ----------
    freq
        Frequency of the data to train on and predict
    prediction_length
        Length of the prediction horizon
    target_dim
        Dimensionality of the input dataset
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
    num_parallel_samples
        Number of evaluation samples per time series to increase parallelism
        during inference. This is a model optimization that does not affect
        the accuracy (default: 100)
    dropout_rate
        Dropout regularization parameter (default: 0.1)
    target_dim_sample
        Number of dimensions to sample for the GP model
    distr_output
        Distribution to use to evaluate observations and sample predictions
        (default: LowrankGPOutput with dim=target_dim and
        rank=5). Note that target dim of the DistributionOutput and the
        estimator constructor call need to match. Also note that the rank in
        this constructor is meaningless if the DistributionOutput is
        constructed outside of this class.
    rank
        Rank for the LowrankGPOutput. (default: 2)
    scaling
        Whether to automatically scale the target values (default: true)
    pick_incomplete
        Whether training examples can be sampled with only a part of
        past_length time-units
    lags_seq
        Indices of the lagged target values to use as inputs of the RNN
        (default: None, in which case these are automatically determined
        based on freq)
    shuffle_target_dim
        Shuffle the dimensions before sampling.
    time_features
        Time features to use as inputs of the RNN (default: None, in which
        case these are automatically determined based on freq)
    conditioning_length
        Set maximum length for conditioning the marginal transformation
    use_marginal_transformation
        Whether marginal (CDFtoGaussianTransform) transformation is used by the
        model
    train_sampler
        Controls the sampling of windows during training.
    validation_sampler
        Controls the sampling of windows during validation.
    batch_size
        The size of the batches to be used training and prediction.
    """

    @validated()
    def __init__(
        self,
        freq: str,
        prediction_length: int,
        target_dim: int,
        trainer: Trainer = Trainer(),
        # number of dimension to sample at training time
        context_length: Optional[int] = None,
        num_layers: int = 2,
        num_cells: int = 40,
        cell_type: str = "lstm",
        num_parallel_samples: int = 100,
        dropout_rate: float = 0.1,
        target_dim_sample: Optional[int] = None,
        distr_output: Optional[DistributionOutput] = None,
        rank: Optional[int] = 2,
        scaling: bool = True,
        pick_incomplete: bool = False,
        lags_seq: Optional[List[int]] = None,
        shuffle_target_dim: bool = True,
        time_features: Optional[List[TimeFeature]] = None,
        conditioning_length: int = 100,
        use_marginal_transformation: bool = False,
        train_sampler: Optional[InstanceSampler] = None,
        validation_sampler: Optional[InstanceSampler] = None,
        batch_size: int = 32,
    ) -> None:
        super().__init__(trainer=trainer, batch_size=batch_size)

        assert (
            prediction_length > 0
        ), "The value of `prediction_length` should be > 0"
        assert (
            context_length is None or context_length > 0
        ), "The value of `context_length` should be > 0"
        assert num_layers > 0, "The value of `num_layers` should be > 0"
        assert num_cells > 0, "The value of `num_cells` should be > 0"
        assert (
            num_parallel_samples > 0
        ), "The value of `num_eval_samples` should be > 0"
        assert dropout_rate >= 0, "The value of `dropout_rate` should be >= 0"

        if distr_output is not None:
            self.distr_output = distr_output
        else:
            self.distr_output = LowrankGPOutput(rank=rank)
        self.freq = freq
        self.context_length = (
            context_length if context_length is not None else prediction_length
        )
        self.prediction_length = prediction_length
        self.target_dim = target_dim
        self.target_dim_sample = (
            target_dim
            if target_dim_sample is None
            else min(target_dim_sample, target_dim)
        )
        self.shuffle_target_dim = shuffle_target_dim
        self.num_layers = num_layers
        self.num_cells = num_cells
        self.cell_type = cell_type
        self.num_parallel_samples = num_parallel_samples
        self.dropout_rate = dropout_rate

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
        self.pick_incomplete = pick_incomplete
        self.scaling = scaling
        self.conditioning_length = conditioning_length
        self.use_marginal_transformation = use_marginal_transformation
        self.output_transform = (
            cdf_to_gaussian_forward_transform
            if self.use_marginal_transformation
            else None
        )

        self.train_sampler = (
            train_sampler
            if train_sampler is not None
            else ExpectedNumInstanceSampler(
                num_instances=1.0,
                min_past=0 if pick_incomplete else self.history_length,
                min_future=prediction_length,
            )
        )
        self.validation_sampler = (
            validation_sampler
            if validation_sampler is not None
            else ValidationSplitSampler(
                min_past=0 if pick_incomplete else self.history_length,
                min_future=prediction_length,
            )
        )

    def create_transformation(self) -> Transformation:
        return Chain(
            [
                AsNumpyArray(
                    field=FieldName.TARGET,
                    expected_ndim=1 + len(self.distr_output.event_shape),
                ),
                # maps the target to (1, T) if the target data is uni
                # dimensional
                ExpandDimArray(
                    field=FieldName.TARGET,
                    axis=0 if self.distr_output.event_shape[0] == 1 else None,
                ),
                AddObservedValuesIndicator(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.OBSERVED_VALUES,
                ),
                AddTimeFeatures(
                    start_field=FieldName.START,
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_TIME,
                    time_features=self.time_features,
                    pred_length=self.prediction_length,
                ),
                VstackFeatures(
                    output_field=FieldName.FEAT_TIME,
                    input_fields=[FieldName.FEAT_TIME],
                ),
                SetFieldIfNotPresent(
                    field=FieldName.FEAT_STATIC_CAT, value=[0.0]
                ),
                TargetDimIndicator(
                    field_name=FieldName.TARGET_DIM_INDICATOR,
                    target_field=FieldName.TARGET,
                ),
                AsNumpyArray(field=FieldName.FEAT_STATIC_CAT, expected_ndim=1),
            ]
        )

    def _create_instance_splitter(self, mode: str):
        assert mode in ["training", "validation", "test"]

        instance_sampler = {
            "training": self.train_sampler,
            "validation": self.validation_sampler,
            "test": TestSplitSampler(),
        }[mode]

        return (
            InstanceSplitter(
                target_field=FieldName.TARGET,
                is_pad_field=FieldName.IS_PAD,
                start_field=FieldName.START,
                forecast_start_field=FieldName.FORECAST_START,
                instance_sampler=instance_sampler,
                past_length=self.history_length,
                future_length=self.prediction_length,
                time_series_fields=[
                    FieldName.FEAT_TIME,
                    FieldName.OBSERVED_VALUES,
                ],
            )
            + (
                CDFtoGaussianTransform(
                    target_field=FieldName.TARGET,
                    observed_values_field=FieldName.OBSERVED_VALUES,
                    max_context_length=self.conditioning_length,
                    target_dim=self.target_dim,
                )
                if self.use_marginal_transformation
                else RenameFields(
                    {
                        f"past_{FieldName.TARGET}": f"past_{FieldName.TARGET}_cdf",
                        f"future_{FieldName.TARGET}": f"future_{FieldName.TARGET}_cdf",
                    }
                )
            )
            + SampleTargetDim(
                field_name=FieldName.TARGET_DIM_INDICATOR,
                target_field=FieldName.TARGET + "_cdf",
                observed_values_field=FieldName.OBSERVED_VALUES,
                num_samples=self.target_dim_sample,
                shuffle=self.shuffle_target_dim,
            )
        )

    def create_training_data_loader(
        self,
        data: Dataset,
        **kwargs,
    ) -> DataLoader:
        input_names = get_hybrid_forward_input_names(GPVARTrainingNetwork)
        instance_splitter = self._create_instance_splitter("training")
        return TrainDataLoader(
            dataset=data,
            transform=instance_splitter + SelectFields(input_names),
            batch_size=self.batch_size,
            stack_fn=partial(batchify, ctx=self.trainer.ctx, dtype=self.dtype),
            decode_fn=partial(as_in_context, ctx=self.trainer.ctx),
            **kwargs,
        )

    def create_validation_data_loader(
        self,
        data: Dataset,
        **kwargs,
    ) -> DataLoader:
        input_names = get_hybrid_forward_input_names(GPVARTrainingNetwork)
        instance_splitter = self._create_instance_splitter("validation")
        return ValidationDataLoader(
            dataset=data,
            transform=instance_splitter + SelectFields(input_names),
            batch_size=self.batch_size,
            stack_fn=partial(batchify, ctx=self.trainer.ctx, dtype=self.dtype),
        )

    def create_training_network(self) -> GPVARTrainingNetwork:
        return GPVARTrainingNetwork(
            target_dim=self.target_dim,
            target_dim_sample=self.target_dim_sample,
            num_layers=self.num_layers,
            num_cells=self.num_cells,
            cell_type=self.cell_type,
            history_length=self.history_length,
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            dropout_rate=self.dropout_rate,
            lags_seq=self.lags_seq,
            scaling=self.scaling,
            distr_output=self.distr_output,
            conditioning_length=self.conditioning_length,
        )

    def create_predictor(
        self, transformation: Transformation, trained_network: HybridBlock
    ) -> Predictor:
        prediction_splitter = self._create_instance_splitter("test")

        prediction_network = GPVARPredictionNetwork(
            target_dim=self.target_dim,
            target_dim_sample=self.target_dim,
            num_parallel_samples=self.num_parallel_samples,
            num_layers=self.num_layers,
            num_cells=self.num_cells,
            cell_type=self.cell_type,
            history_length=self.history_length,
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            dropout_rate=self.dropout_rate,
            lags_seq=self.lags_seq,
            scaling=self.scaling,
            distr_output=self.distr_output,
            conditioning_length=self.conditioning_length,
        )

        copy_parameters(trained_network, prediction_network)

        return RepresentableBlockPredictor(
            input_transform=transformation + prediction_splitter,
            prediction_net=prediction_network,
            batch_size=self.batch_size,
            freq=self.freq,
            prediction_length=self.prediction_length,
            ctx=self.trainer.ctx,
            output_transform=self.output_transform,
        )
