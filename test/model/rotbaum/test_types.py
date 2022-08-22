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

import pytest
from pydantic import ValidationError

from gluonts.model.rotbaum._types import (
    FeatureImportanceResult,
    ExplanationResult,
)


@pytest.mark.parametrize(
    "feature_importance_results_match",
    [
        {
            "target": [1.0],
            "feat_static_cat": [1.0],
            "feat_static_real": [1.0],
            "past_feat_dynamic_real": [1.0],
            "feat_dynamic_real": [1.0],
            "feat_dynamic_cat": [1.0],
        },
        {
            "target": [1.0],
            "feat_static_cat": [],
            "feat_static_real": [],
            "past_feat_dynamic_real": [],
            "feat_dynamic_real": [],
            "feat_dynamic_cat": [],
        },
        {
            "target": [[1.0]],
            "feat_static_cat": [[1.0]],
            "feat_static_real": [[1.0]],
            "past_feat_dynamic_real": [[1.0]],
            "feat_dynamic_real": [[1.0]],
            "feat_dynamic_cat": [[1.0]],
        },
        {
            "target": [[1.0]],
            "feat_static_cat": [],
            "feat_static_real": [],
            "past_feat_dynamic_real": [],
            "feat_dynamic_real": [],
            "feat_dynamic_cat": [],
        },
    ],
)
def test_feature_importance_shape_match(feature_importance_results_match):
    FeatureImportanceResult(**feature_importance_results_match)


@pytest.mark.parametrize(
    "feature_importance_results_mismatch",
    [
        {
            "target": [[[]]],
            "feat_static_cat": [1.0],
            "feat_static_real": [1.0],
            "past_feat_dynamic_real": [1.0],
            "feat_dynamic_real": [1.0],
            "feat_dynamic_cat": [1.0],
        },
        {
            "target": [[1.0, 1.0]],
            "feat_static_cat": [[1.0]],
            "feat_static_real": [[1.0]],
            "past_feat_dynamic_real": [[1.0]],
            "feat_dynamic_real": [[1.0]],
            "feat_dynamic_cat": [[1.0]],
        },
        {
            "target": [[1.0]],
            "feat_static_cat": [[1.0, 1.0]],
            "feat_static_real": [],
            "past_feat_dynamic_real": [],
            "feat_dynamic_real": [],
            "feat_dynamic_cat": [],
        },
    ],
)
def test_feature_importance_shape_match(feature_importance_results_mismatch):
    with pytest.raises(ValidationError) as ve:
        FeatureImportanceResult(**feature_importance_results_mismatch)


def test_explanation_result():
    data_dict = {
        "target": [[1.0, 2.0]],
        "feat_static_cat": [[1.0, 2.0], [1.0, 2.0]],
        "feat_static_real": [[1.0, 2.0], [1.0, 2.0]],
        "past_feat_dynamic_real": [[1.0, 2.0], [1.0, 2.0]],
        "feat_dynamic_real": [[1.0, 2.0], [1.0, 2.0]],
        "feat_dynamic_cat": [[1.0, 2.0], [1.0, 2.0]],
    }

    assert (
        ExplanationResult(
            quantile_aggregated_result=FeatureImportanceResult(**data_dict),
            time_quantile_aggregated_result=FeatureImportanceResult(
                **data_dict
            ).mean(axis=1),
        ).time_quantile_aggregated_result
        is not None
    )
    assert (
        ExplanationResult(
            quantile_aggregated_result=FeatureImportanceResult(**data_dict),
        ).time_quantile_aggregated_result
        is None
    )

    with pytest.raises(ValidationError) as ve:
        ExplanationResult()
