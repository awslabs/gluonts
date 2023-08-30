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

from typing import Dict, Union
import pandas as pd
from tsbench.config import Config, ModelConfig
from tsbench.evaluations.metrics import Performance
from tsbench.evaluations.tracking import ModelTracker


class ConfigAnalyzer:
    """
    The config analyzer evaluates a single model configuration by obtaining its
    performance from a tracker across datasets and compiling it into a
    dataframe.
    """

    def __init__(self, tracker: ModelTracker):
        """
        Args:
            tracker: The tracker which is used to obtain performances.
        """
        self.tracker = tracker
        self.datasets = {
            c.dataset for c in self.tracker.get_evaluations().configurations
        }

    def run(
        self, model_config: Union[ModelConfig, Dict[str, ModelConfig]]
    ) -> pd.DataFrame:
        """
        Runs the evaluation, providing a configuration's performances for all
        datasets.

        Args:
            model_config: A single configuration for which to obtain performances or a mapping from
                dataset names to model configurations.

        Returns:
            The metrics on individual datasets.
        """
        results = []
        for dataset in self.datasets:
            # Construct the config
            if isinstance(model_config, dict):
                config = Config(model_config[dataset.name()], dataset)
            else:
                config = Config(model_config, dataset)

            # Get the performance and append to results
            performance = self.tracker.get_performance(config)
            df = Performance.to_dataframe([performance]).assign(
                test_dataset=dataset.name()
            )
            results.append(df)

        return pd.concat(results).set_index("test_dataset")
