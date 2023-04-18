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

import numpy as np
from mxnet.gluon import HybridBlock

from gluonts.core.component import Type, validated
from gluonts.dataset.common import Dataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.loader import (
    DataLoader,
    TrainDataLoader,
    ValidationDataLoader,
)
from gluonts.dataset.stat import calculate_dataset_statistics
from gluonts.env import env
from gluonts.model.predictor import Predictor
from gluonts.mx.batchify import as_in_context, batchify
from gluonts.mx.distribution import DistributionOutput, StudentTOutput
from gluonts.mx.model.estimator import GluonEstimator
from gluonts.mx.model.predictor import RepresentableBlockPredictor
from gluonts.mx.trainer import Trainer
from gluonts.mx.util import copy_parameters, get_hybrid_forward_input_names
from gluonts.itertools import maybe_len
from gluonts.time_feature import (
    TimeFeature,
    get_lags_for_frequency,
    time_features_from_frequency_str,
)
from gluonts.transform import (
    AddAgeFeature,
    AddObservedValuesIndicator,
    AddTimeFeatures,
    AsNumpyArray,
    Chain,
    ExpectedNumInstanceSampler,
    InstanceSampler,
    InstanceSplitter,
    RemoveFields,
    SelectFields,
    SetField,
    TestSplitSampler,
    Transformation,
    ValidationSplitSampler,
    VstackFeatures,
)
from gluonts.transform.feature import (
    DummyValueImputation,
    MissingValueImputation,
)

from gluonts.mx.model.deepar import DeepAREstimator


# Hack to expose history_length to constructor to allow for proper
# serialization
class DeepAREstimatorForCOP(DeepAREstimator):
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
        dropoutcell_type: str = "ZoneoutCell",
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
        imputation_method: Optional[MissingValueImputation] = None,
        train_sampler: Optional[InstanceSampler] = None,
        validation_sampler: Optional[InstanceSampler] = None,
        dtype: Type = np.float32,
        alpha: float = 0.0,
        beta: float = 0.0,
        batch_size: int = 32,
        default_scale: Optional[float] = None,
        minimum_scale: float = 1e-10,
        impute_missing_values: bool = False,
        num_imputation_samples: int = 1,
        history_length: Optional[int] = None,
    ) -> None:
        super().__init__(
            freq=freq,
            prediction_length=prediction_length,
            trainer=trainer,
            context_length=context_length,
            num_layers=num_layers,
            num_cells=num_cells,
            cell_type=cell_type,
            dropoutcell_type=dropoutcell_type,
            dropout_rate=dropout_rate,
            use_feat_dynamic_real=use_feat_dynamic_real,
            use_feat_static_cat=use_feat_static_cat,
            use_feat_static_real=use_feat_static_real,
            cardinality=cardinality,
            embedding_dimension=embedding_dimension,
            distr_output=distr_output,
            scaling=scaling,
            lags_seq=lags_seq,
            time_features=time_features,
            num_parallel_samples=num_parallel_samples,
            imputation_method=imputation_method,
            train_sampler=train_sampler,
            validation_sampler=validation_sampler,
            alpha=alpha,
            beta=beta,
            default_scale=default_scale,
            minimum_scale=minimum_scale,
            impute_missing_values=impute_missing_values,
            num_imputation_samples=num_imputation_samples,
            batch_size=batch_size,
            dtype=dtype,
        )
        self.freq = freq
        self.history_length = (
            history_length
            if history_length is not None
            else self.history_length
        )
