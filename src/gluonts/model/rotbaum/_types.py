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

from typing import List, Union, Optional

import numpy as np
from pydantic import BaseModel, root_validator


class FeatureImportanceResult(BaseModel):
    target: List[Union[List[float], float]]
    feat_static_cat: List[Union[List[float], float]]
    feat_static_real: List[Union[List[float], float]]
    past_feat_dynamic_real: List[Union[List[float], float]]
    feat_dynamic_real: List[Union[List[float], float]]
    feat_dynamic_cat: List[Union[List[float], float]]

    @root_validator()
    def check_shape(cls, values):
        """
        Validate the second dimension is the same for 2d results and all fields share the same dimensionality
        For example, time aligned results with dimension of (features, pred_length), the pred_length shall be the same
        :param values:
        :return:
        """
        dim = np.array(values.get("target")).ndim
        assert (
            0 < dim <= 2
        ), "expected the feature importances array to be in the dimension of 1d or 2d only but got {dim}d from target"
        for key, value in values.items():
            if value:
                assert (
                    np.array(value).ndim == dim
                ), f"dimension mismatch {key} with dim {np.array(value).ndim} and target with dim {dim} "
        if dim == 1:
            return values
        shape = np.shape(values.get("target"))[dim - 1]
        for key, value in values.items():
            if value:
                assert (
                    np.shape(value)[dim - 1] == shape
                ), f"shape mismatch {key} with shape {np.shape(value)} and target with shape {shape} "
        return values

    def mean(self, axis=None) -> "FeatureImportanceResult":
        mean_dict = {}
        for key, val in self.dict().items():
            mean_dict[key] = np.mean(val, axis=axis).tolist() if val else []
        return FeatureImportanceResult(**mean_dict)


class ExplanationResult(BaseModel):
    time_quantile_aggregated_result: Optional[FeatureImportanceResult]
    quantile_aggregated_result: FeatureImportanceResult
