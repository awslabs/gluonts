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

from gluonts.dataset.field_names import FieldName


def test_dataset_fields():
    assert (
        "feat_static_cat" == FieldName.FEAT_STATIC_CAT
    ), "Error in the FieldName 'feat_static_cat'."
    assert (
        "feat_static_real" == FieldName.FEAT_STATIC_REAL
    ), "Error in the FieldName 'feat_static_real'."
    assert (
        "feat_dynamic_cat" == FieldName.FEAT_DYNAMIC_CAT
    ), "Error in the FieldName 'feat_dynamic_cat'."
    assert (
        "feat_dynamic_real" == FieldName.FEAT_DYNAMIC_REAL
    ), "Error in the FieldName 'feat_dynamic_real'."
