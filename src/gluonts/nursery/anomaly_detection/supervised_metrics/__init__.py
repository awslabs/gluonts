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

from ._buffered_precision_recall import buffered_precision_recall
from ._precision_recall_utils import (
    aggregate_precision_recall,
    aggregate_precision_recall_curve,
)
from ._segment_precision_recall import segment_precision_recall

__all__ = [
    "aggregate_precision_recall",
    "aggregate_precision_recall_curve",
    "buffered_precision_recall",
    "segment_precision_recall",
]

# fix Sphinx issues, see https://bit.ly/2K2eptM
for item in __all__:
    if hasattr(item, "__module__"):
        setattr(item, "__module__", __name__)
