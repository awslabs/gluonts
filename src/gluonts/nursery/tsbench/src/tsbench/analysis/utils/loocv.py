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

from typing import Iterator, List, Tuple, TypeVar
from sklearn.model_selection import LeaveOneGroupOut
from tsbench.config import Config, EnsembleConfig, ModelConfig
from tsbench.evaluations.metrics import Performance
from tsbench.evaluations.tracking import Tracker

T = TypeVar("T", ModelConfig, EnsembleConfig)


def loocv_split(
    tracker: Tracker[T],
) -> Iterator[
    Tuple[
        Tuple[List[Config[T]], List[Performance]],
        Tuple[List[Config[T]], List[Performance]],
    ]
]:
    """
    Iterates over the configurations and associated performances obtained from
    a collector. For each item it yields, it leaves out
    configurations/performances for a single dataset for testing and provides
    the configurations/performances for training.

    Args:
        tracker: The tracker to retrieve data from.
        show_progerss: Whether to show progress via tqdm.

    Yields:
        The training data, including configurations and performances, and the test data, including
        configurations and performances.
    """
    data = tracker.get_evaluations()
    groups = [c.dataset.name() for c in data.configurations]

    # Split the data according to the datasets
    loocv = LeaveOneGroupOut()
    for I_train, I_test in loocv.split(data.configurations, groups=groups):
        X_train = [data.configurations[i] for i in I_train]
        y_train = [data.performances[i] for i in I_train]

        X_test = [data.configurations[i] for i in I_test]
        y_test = [data.performances[i] for i in I_test]

        yield (X_train, y_train), (X_test, y_test)
