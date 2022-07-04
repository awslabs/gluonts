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

import warnings

warnings.warn(
    "The san model in gulonts.model is deprecated and will be moved to "
    "'gluonts.mx.model'. Try to use 'from gluonts.mx import "
    "SelfAttentionEstimator'.",
    FutureWarning,
)

from gluonts.mx.model.san._estimator import SelfAttentionEstimator

__all__ = [
    "SelfAttentionEstimator",
]
