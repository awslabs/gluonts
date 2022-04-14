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

from pathlib import Path
import yaml
from black import click
from cli.utils import (
    explode_key_values,
    iterate_configurations,
    run_sacred_script,
)
from tsbench.constants import DEFAULT_DATA_PATH, DEFAULT_EVALUATIONS_PATH
from ._main import analysis


@analysis.command(short_help="Evaluate the ranking performance of surrogates.")
@click.option(
    "--experiment",
    required=True,
    help=(
        "The name of the experiment under which the individual training runs"
        " are grouped."
    ),
)
@click.option(
    "--config_path",
    required=True,
    help="The local path to the configuration file defining script options.",
)
@click.option(
    "--data_path",
    type=click.Path(exists=True),
    default=DEFAULT_DATA_PATH,
    show_default=True,
    help="The path where datasets are stored.",
)
@click.option(
    "--evaluations_path",
    type=click.Path(exists=True),
    default=DEFAULT_EVALUATIONS_PATH,
    show_default=True,
    help="The path where offline evaluations are stored.",
)
@click.option(
    "--nskip",
    default=0,
    show_default=True,
    help=(
        "The number of configurations to skip. Useful if some set of"
        " experiments failed."
    ),
)
def surrogate(
    experiment: str,
    config_path: str,
    data_path: str,
    evaluations_path: str,
    nskip: int,
):
    """
    Evaluates the performance of a set of surrogate models using the available
    offline evaluations. Performance is evaluated via ranking metrics and
    performed via stratified leave-one-out cross-validation where each stratum
    consists of the evaluations on a single evaluation dataset.

    This call runs the Sacred script for each provided configuration
    sequentially and returns only once all runs have completed.
    """
    with Path(config_path).open("r", encoding="utf-8") as f:
        content = yaml.safe_load(f)
        configs = explode_key_values("surrogate", content)

    for configuration in iterate_configurations(configs, nskip):
        run_sacred_script(
            "surrogate.py",
            experiment=experiment,
            data_path=data_path,
            evaluations_path=evaluations_path,
            **configuration,
        )
