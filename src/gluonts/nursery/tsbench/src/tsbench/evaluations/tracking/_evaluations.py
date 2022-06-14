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
from typing import cast, Generic, List, TypeVar
import pandas as pd
from tsbench.config import Config, EnsembleConfig, ModelConfig
from tsbench.evaluations.metrics import Performance

T = TypeVar("T", ModelConfig, EnsembleConfig)


@dataclass
class Evaluations(Generic[T]):
    """
    Generic base for evaluations of models.
    """

    configurations: List[Config[T]]
    performances: List[Performance]

    def dataframe(self, std: bool = True) -> pd.DataFrame:
        """
        Returns a dataframe which contains the performance metrics as columns
        and the configurations as multi-index.

        Args:
            std: Whether to include the standard deviation of performance metrics in the dataframe.
        """
        # Should implement this for ensembles as well
        index_df = Config.to_dataframe(
            cast(List[Config[ModelConfig]], self.configurations)
        )
        # Reorder columns
        column_order = ["dataset"] + [
            c for c in index_df.columns.tolist() if c != "dataset"
        ]
        index = pd.MultiIndex.from_frame(index_df[column_order])
        df = Performance.to_dataframe(self.performances, std=std)
        df.index = index
        return df.sort_index()
