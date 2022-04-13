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
import click
from tsbench.constants import DEFAULT_EVALUATIONS_PATH
from tsbench.utils import compress_directory
from ._main import evaluations


@evaluations.command(
    short_help="Archive metrics of all evaluations into a single file."
)
@click.option(
    "--evaluations_path",
    type=click.Path(exists=True),
    default=DEFAULT_EVALUATIONS_PATH,
    help="The directory where TSBench evaluations are stored.",
)
@click.option(
    "--archive_path",
    type=click.Path(),
    default=Path.home() / "archive",
    help="The directory to which to write the compressed files.",
)
def archive(evaluations_path: str, archive_path: str):
    """
    Archives the metrics of all evaluations found in the provided directory
    into a single file.

    This is most probably only necessary for publishing the metrics in an
    easier format.
    """
    source = Path(evaluations_path)
    target = Path(archive_path)
    target.mkdir(parents=True, exist_ok=True)

    # First, tar all the metadata
    print("Compressing metrics...")
    compress_directory(
        source,
        target / "metrics.tar.gz",
        include={"config.json", "performance.json"},
    )
