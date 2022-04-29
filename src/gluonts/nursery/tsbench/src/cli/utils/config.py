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

import itertools
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Literal, TextIO
import yaml
from tsbench.config import TrainConfig


def iterate_configurations(
    configs: List[Dict[str, Any]], skip: int = 0
) -> Iterator[Dict[str, Any]]:
    """
    Provides an iterator over the configurations, ignoring the first `skip`
    configurations.

    Args:
        configs: The configurations to iterate over.
        skip: The number of configurations to skip at the beginning.

    Yields:
        The configuration.
    """
    for i, config_item in enumerate(configs):
        if i < skip:
            print(f">>> Skipping configuration {i+1}/{len(configs)}")
            continue
        print(f">>> Running configuration {i+1}/{len(configs)}")
        yield config_item


def generate_configurations(path: Path) -> List[Any]:
    """
    Generates the hyperparameter configuration for the configuration(s) found
    at the specified path.

    Args:
        path: Either a file or directory from to read the configuration(s). If a directory is
            given, all YAML files are read recursively.

    Returns:
        The configurations.
    """
    if path.is_file():
        with path.open() as f:
            return _generate_configurations(f)

    all_configurations = []
    for file in path.glob("**/*.yaml"):
        with file.open() as f:
            all_configurations.extend(_generate_configurations(f))
    return all_configurations


def explode_key_values(
    primary_key: str,
    mapping: Dict[str, List[Dict[Literal["key", "values"], Any]]],
    process_key: Callable[[str, str], str] = lambda _, s: s,
) -> List[Dict[str, Any]]:
    """
    Explodes a mapping from primary keys to a list of key to value mappings
    into independent configurations.
    """
    all_combinations = {
        primary: itertools.product(
            *[
                [(option["key"], value) for value in option["values"]]
                for option in choices
            ]
        )
        if choices
        else []
        for primary, choices in mapping.items()
    }

    # Generate configs
    configs = []
    for primary, combinations in all_combinations.items():
        if not combinations:
            configs.append({primary_key: primary})
        for item in combinations:
            primary_config = {primary_key: primary}
            for key, value in item:
                if isinstance(key, (list, tuple)):
                    primary_config.update(
                        {
                            process_key(primary, k): v
                            for k, v in zip(key, value)
                        }
                    )
                else:
                    primary_config[process_key(primary, key)] = value
            configs.append(primary_config)

    # Explode repetitions
    result = []
    for config in configs:
        if "__repeat__" in config:
            for _ in range(config["__repeat__"]):
                result.append(
                    {k: v for k, v in config.items() if k != "__repeat__"}
                )
        else:
            result.append(config)

    return result


def _generate_configurations(config: TextIO) -> List[Any]:
    # First, we parse the config file
    options = yaml.safe_load(config)
    seeds = options["seeds"]
    datasets = options["datasets"]

    def process_key(model: str, key: str) -> str:
        if key in TrainConfig.training_hyperparameters():
            return key
        return f"{model}_{key}"

    configs = explode_key_values("model", options["models"], process_key)

    # Explode all runs and save to file
    all_configurations = []
    for seed in seeds:
        for dataset in datasets:
            for model_config in configs:
                all_configurations.append(
                    {
                        "seed": seed,
                        "dataset": dataset,
                        **model_config,
                    }
                )

    return all_configurations
