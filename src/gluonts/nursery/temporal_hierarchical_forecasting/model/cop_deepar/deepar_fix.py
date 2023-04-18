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

from typing import List, Optional

import numpy as np

from gluonts.core.component import Type, validated
from gluonts.mx.distribution import DistributionOutput, StudentTOutput
from gluonts.mx.trainer import Trainer
from gluonts.time_feature import TimeFeature
from gluonts.transform import InstanceSampler
from gluonts.transform.feature import MissingValueImputation

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
        self.history_length: int = (
            history_length
            if history_length is not None
            else self.history_length
        )
