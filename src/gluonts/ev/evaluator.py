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

from __future__ import annotations

from dataclasses import dataclass
from operator import methodcaller
from typing import Dict, Mapping, Iterator

import numpy as np
from toolz import valmap


@dataclass
class Evaluator:
    metrics: Dict[str, Metric]

    def update(self, data: Mapping[str, np.ndarray]) -> None:
        for metric in self.metrics.values():
            metric.update(data)

    def update_all(self, stream: Iterator[Mapping[str, np.ndarray]]) -> None:
        for element in stream:
            self.update(element)

    def get(self) -> Dict[str, np.ndarray]:
        return valmap(methodcaller("get"), self.metrics)


def evaluate(metrics, data_batches, axis=None):
    evaluator = metrics(axis)
    evaluator.update_all(data_batches)
    return evaluator.get()


from .metrics import Metric  # noqa
