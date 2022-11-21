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

from dataclasses import dataclass
from typing import Callable, ChainMap, Dict

import numpy as np

from .aggregations import Aggregation


@dataclass
class Evaluator:
    name: str

    def update(self, data: ChainMap[str, np.ndarray]) -> None:
        raise NotImplementedError

    def get(self) -> np.ndarray:
        raise NotImplementedError


@dataclass
class DirectEvaluator(Evaluator):
    """An Evaluator which uses a single function and aggregation strategy."""

    stat: Callable
    aggregate: Aggregation

    def update(self, data: ChainMap[str, np.ndarray]) -> None:
        self.aggregate.step(self.stat(data))

    def get(self) -> np.ndarray:
        return self.aggregate.get()


@dataclass
class DerivedEvaluator(Evaluator):
    """An Evaluator for metrics that are derived from other metrics.

    A derived metric updates multiple, simpler metrics independently and in
    the end combines their results as defined in `post_process`."""

    evaluators: Dict[str, Evaluator]
    post_process: Callable

    def update(self, data: ChainMap[str, np.ndarray]) -> None:
        for evaluator in self.evaluators.values():
            evaluator.update(data)

    def get(self) -> np.ndarray:
        return self.post_process(
            **{
                name: evaluator.get()
                for name, evaluator in self.evaluators.items()
            }
        )
