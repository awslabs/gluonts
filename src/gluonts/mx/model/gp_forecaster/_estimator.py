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
from typing import List, Optional, Type

import numpy as np
from mxnet.gluon import HybridBlock

from gluonts.core.component import validated
from gluonts.dataset.common import Dataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.loader import (
    DataLoader,
    TrainDataLoader,
    ValidationDataLoader,
)
from gluonts.model.predictor import Predictor
from gluonts.mx.batchify import batchify
from gluonts.mx.kernels import KernelOutput, RBFKernelOutput
from gluonts.mx.model.estimator import GluonEstimator
from gluonts.mx.model.predictor import RepresentableBlockPredictor
from gluonts.mx.trainer import Trainer
from gluonts.mx.util import copy_parameters, get_hybrid_forward_input_names
from gluonts.time_feature import TimeFeature, time_features_from_frequency_str
from gluonts.transform import (
    AddTimeFeatures,
    AsNumpyArray,
    CanonicalInstanceSplitter,
    SelectFields,
    SetFieldIfNotPresent,
    TestSplitSampler,
    Transformation,
)

from ._network import (
    GaussianProcessPredictionNetwork,
    GaussianProcessTrainingNetwork,
)


class GaussianProcessEstimator(GluonEstimator):
    r"""
    GaussianProcessEstimator shows how to build a local time series model using
    Gaussian Processes (GP).

    Each time series has a GP with its own
    hyper-parameters.  For the radial basis function (RBF) Kernel, the
    learnable hyper-parameters are the amplitude and lengthscale. The periodic
    kernel has those hyper-parameters with an additional learnable frequency
    parameter. The RBFKernel is the default, but either kernel can be used by
    inputting the desired KernelOutput object. The noise sigma in the model is
    another learnable hyper-parameter for both kernels. These parameters are
    fit using an Embedding of the integer time series indices (each time series
    has its set of hyper-parameter that is static in time). The observations
    are the time series values. In this model, the time features are hour of
    the day and day of the week.

    Parameters
    ----------
    freq
        Time series frequency.
    prediction_length
        Prediction length.
    cardinality
        Number of time series.
    trainer
        Trainer instance to be used for model training (default: Trainer()).
    context_length
        Training length (default: None, in which case context_length =
        prediction_length).
    kernel_output
        KernelOutput instance to determine which kernel subclass to be
        instantiated (default: RBFKernelOutput()).
    params_scaling
        Determines whether or not to scale the model parameters
        (default: True).
    float_type
        Determines whether to use single or double precision
        (default: np.float64).
    max_iter_jitter
        Maximum number of iterations for jitter to iteratively make the matrix
        positive definite (default: 10).
    jitter_method
        Iteratively jitter method or use eigenvalue decomposition depending on
        problem size (default: "iter").
    sample_noise
        Boolean to determine whether to add :math:`\sigma^2I` to the predictive
        covariance matrix (default: True).
    time_features
        Time features to use as inputs of the model (default: None, in which
        case these are automatically determined based on the frequency).
    num_parallel_samples
        Number of evaluation samples per time series to increase parallelism
        during inference. This is a model optimization that does not affect the
        accuracy (default: 100).
    batch_size
        The size of the batches to be used training and prediction.
    """

    @validated()
    def __init__(
        self,
        freq: str,
        prediction_length: int,
        cardinality: int,
        trainer: Trainer = Trainer(),
        context_length: Optional[int] = None,
        kernel_output: KernelOutput = RBFKernelOutput(),
        params_scaling: bool = True,
        dtype: Type = np.float64,
        max_iter_jitter: int = 10,
        jitter_method: str = "iter",
        sample_noise: bool = True,
        time_features: Optional[List[TimeFeature]] = None,
        num_parallel_samples: int = 100,
        batch_size: int = 32,
    ) -> None:
        self.float_type = dtype
        super().__init__(
            trainer=trainer, batch_size=batch_size, dtype=self.float_type
        )

        assert (
            prediction_length > 0
        ), "The value of `prediction_length` should be > 0"
        assert cardinality > 0, "The value of `cardinality` should be > 0"
        assert (
            context_length is None or context_length > 0
        ), "The value of `context_length` should be > 0"
        assert (
            num_parallel_samples > 0
        ), "The value of `num_parallel_samples` should be > 0"

        self.prediction_length = prediction_length
        self.context_length = (
            context_length if context_length is not None else prediction_length
        )
        self.cardinality = cardinality
        self.kernel_output = kernel_output
        self.params_scaling = params_scaling
        self.max_iter_jitter = max_iter_jitter
        self.jitter_method = jitter_method
        self.sample_noise = sample_noise
        self.time_features = (
            time_features
            if time_features is not None
            else time_features_from_frequency_str(freq)
        )
        self.num_parallel_samples = num_parallel_samples

    def create_transformation(self) -> Transformation:
        return (
            AsNumpyArray(field=FieldName.TARGET, expected_ndim=1)
            + AddTimeFeatures(
                start_field=FieldName.START,
                target_field=FieldName.TARGET,
                output_field=FieldName.FEAT_TIME,
                time_features=self.time_features,
                pred_length=self.prediction_length,
            )
            + SetFieldIfNotPresent(
                field=FieldName.FEAT_STATIC_CAT, value=[0.0]
            )
            + AsNumpyArray(field=FieldName.FEAT_STATIC_CAT, expected_ndim=1)
        )

    def _create_instance_splitter(self, mode: str):
        assert mode in ["training", "validation", "test"]

        return CanonicalInstanceSplitter(
            target_field=FieldName.TARGET,
            is_pad_field=FieldName.IS_PAD,
            start_field=FieldName.START,
            forecast_start_field=FieldName.FORECAST_START,
            instance_sampler=TestSplitSampler(),
            time_series_fields=[FieldName.FEAT_TIME],
            instance_length=self.context_length,
            use_prediction_features=(mode != "training"),
            prediction_length=self.prediction_length,
        )

    def create_training_data_loader(
        self,
        data: Dataset,
        **kwargs,
    ) -> DataLoader:
        input_names = get_hybrid_forward_input_names(
            GaussianProcessTrainingNetwork
        )
        instance_splitter = self._create_instance_splitter("training")
        return TrainDataLoader(
            dataset=data,
            transform=instance_splitter + SelectFields(input_names),
            batch_size=self.batch_size,
            stack_fn=partial(batchify, ctx=self.trainer.ctx, dtype=self.dtype),
            **kwargs,
        )

    def create_validation_data_loader(
        self,
        data: Dataset,
        **kwargs,
    ) -> DataLoader:
        input_names = get_hybrid_forward_input_names(
            GaussianProcessTrainingNetwork
        )
        instance_splitter = self._create_instance_splitter("validation")
        return ValidationDataLoader(
            dataset=data,
            transform=instance_splitter + SelectFields(input_names),
            batch_size=self.batch_size,
            stack_fn=partial(batchify, ctx=self.trainer.ctx, dtype=self.dtype),
        )

    def create_training_network(self) -> HybridBlock:
        return GaussianProcessTrainingNetwork(
            prediction_length=self.prediction_length,
            context_length=self.context_length,
            cardinality=self.cardinality,
            kernel_output=self.kernel_output,
            params_scaling=self.params_scaling,
            float_type=self.float_type,
            max_iter_jitter=self.max_iter_jitter,
            jitter_method=self.jitter_method,
        )

    def create_predictor(
        self, transformation: Transformation, trained_network: HybridBlock
    ) -> Predictor:
        prediction_splitter = self._create_instance_splitter("test")

        prediction_network = GaussianProcessPredictionNetwork(
            prediction_length=self.prediction_length,
            context_length=self.context_length,
            cardinality=self.cardinality,
            num_parallel_samples=self.num_parallel_samples,
            params=trained_network.collect_params(),
            kernel_output=self.kernel_output,
            params_scaling=self.params_scaling,
            float_type=self.float_type,
            max_iter_jitter=self.max_iter_jitter,
            jitter_method=self.jitter_method,
            sample_noise=self.sample_noise,
        )

        copy_parameters(
            net_source=trained_network, net_dest=prediction_network
        )

        return RepresentableBlockPredictor(
            input_transform=transformation + prediction_splitter,
            prediction_net=prediction_network,
            batch_size=self.batch_size,
            prediction_length=self.prediction_length,
            ctx=self.trainer.ctx,
            dtype=self.float_type,
        )
