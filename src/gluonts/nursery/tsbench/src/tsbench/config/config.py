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
from dataclasses import asdict, dataclass
from typing import Dict, FrozenSet, Generic, List, TypeVar, Union
import pandas as pd
from .dataset import DatasetConfig
from .model import ModelConfig, TrainConfig

EnsembleConfig = FrozenSet[ModelConfig]

T = TypeVar("T", ModelConfig, EnsembleConfig)


@dataclass(frozen=True)
class Config(Generic[T]):
    """
    A configuration is a tuple of a model configuration and a dataset
    configuration.
    """

    @classmethod
    def to_dataframe(
        cls,
        configs: list[Config[ModelConfig]],
    ) -> pd.DataFrame:
        """
        Returns a data frame representing the provided configurations.

        The model is translated into columns of the data frame automatically.
        The dataset is converted to a single column by being represented by its
        name.
        """
        rows: list[dict[str, str | float | int | bool]] = [
            {
                "model": c.model.name(),
                "dataset": c.dataset.name(),
                **{
                    (
                        f"model_{k}"
                        if k in TrainConfig.training_hyperparameters()
                        else f"model_{c.model.name()}_{k}"
                    ): v
                    for k, v in asdict(c.model).items()
                },
            }
            for c in configs
        ]
        return pd.DataFrame(rows)

    model: T
    dataset: DatasetConfig
