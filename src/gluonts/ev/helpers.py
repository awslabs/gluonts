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

from typing import (
    Collection,
    Iterable,
    Optional,
)

import numpy as np

from .metrics import Metric


def evaluate(
    metrics: Collection[Metric],
    data_batches: Iterable[np.ndarray],
    axis: Optional[int] = None,
):
    evaluators = {}
    for metric in metrics:
        evaluator = metric(axis=axis)

        assert (
            evaluator.name not in evaluators
        ), f"Evaluator name '{evaluator.name}' is not unique."

        evaluators[evaluator.name] = evaluator

    for data_batch in iter(data_batches):
        for evaluator in evaluators.values():
            evaluator.update(data_batch)

    return {
        metric_name: evaluator.get()
        for metric_name, evaluator in evaluators.items()
    }
