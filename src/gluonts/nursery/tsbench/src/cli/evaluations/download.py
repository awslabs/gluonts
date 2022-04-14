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

import os
import tarfile
import tempfile
from functools import partial
from pathlib import Path
from typing import Any, cast, Dict, List, Optional
import botocore
import click
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map
from tsbench.analysis.utils import run_parallel
from tsbench.constants import DEFAULT_EVALUATIONS_PATH
from tsbench.evaluations import aws
from tsbench.evaluations.aws import default_session
from tsbench.evaluations.tracking.job import Job, load_jobs_from_analysis
from ._main import evaluations


@evaluations.command(short_help="Download evaluations to your file system.")
@click.option(
    "--experiment",
    type=str,
    default=None,
    help=(
        "The AWS Sagemaker experiment from which to download the evaluations."
        " If not provided, downloads the publicly available evaluations"
        " (s3://odp-tsbench)."
    ),
)
@click.option(
    "--include_forecasts",
    type=bool,
    default=False,
    help=(
        "Whether to download forecasts (plenty of data) or "
        "only the training, validation and testing metrics."
    ),
)
@click.option(
    "--evaluations_path",
    type=click.Path(),
    default=DEFAULT_EVALUATIONS_PATH,
    show_default=True,
    help="The path to which to download the evaluations to.",
)
def download(
    experiment: Optional[str], include_forecasts: bool, evaluations_path: str
):
    """
    Downloads either the evaluations of a single AWS Sagemaker experiment or
    the publicly available evaluations.

    The evaluations are downloaded to the provided directory.
    """
    target = Path(evaluations_path)
    target.mkdir(parents=True, exist_ok=True)

    if experiment is None:
        print("Downloading publicly available evaluations...")
        _download_public_evaluations(
            include_forecasts=include_forecasts, evaluations_path=target
        )
    else:
        print(f"Downloading data from experiment '{experiment}'...")
        analysis = aws.Analysis(experiment)
        process_map(
            partial(
                _move_job, target=target, include_forecasts=include_forecasts
            ),
            load_jobs_from_analysis(analysis),
            chunksize=1,
        )


def _download_public_evaluations(
    include_forecasts: bool, evaluations_path: Path
) -> None:
    public_bucket = "odp-tsbench"
    session = default_session()
    client = session.client(
        "s3",
        config=botocore.client.Config(  # type: ignore
            signature_version=botocore.UNSIGNED,
            max_pool_connections=2 * cast(int, os.cpu_count()),
        ),
    )

    # First, download the metrics
    print("Downloading metrics...")
    with tempfile.TemporaryDirectory() as tmp:
        file = Path(tmp) / "metrics.tar.gz"
        client.download_file(public_bucket, "metrics.tar.gz", str(file))
        with tarfile.open(file, mode="r:gz") as tar:
            tar.extractall(evaluations_path)

    # Then, optionally download the forecasts
    if include_forecasts:
        print("Downloading forecasts...")

        # First, get all files
        with tqdm(desc="List objects") as progress:
            response = client.list_objects(Bucket=public_bucket)
            objects = _extract_object_names(response)
            progress.update()
            while response["IsTruncated"]:
                response = client.list_objects(
                    Bucket=public_bucket, Marker=objects[-1]
                )
                objects.extend(_extract_object_names(response))
                progress.update()

        # Then, download all of the objects
        run_parallel(
            partial(
                _download_object,
                bucket=public_bucket,
                client=client,
                destination=evaluations_path,
            ),
            objects,
            2 * cast(int, os.cpu_count()),
        )


def _download_object(
    key: str, bucket: str, client: Any, destination: Path
) -> None:
    target = destination / key
    target.parent.mkdir(parents=True, exist_ok=True)
    client.download_file(Bucket=bucket, Key=key, Filename=str(target))


def _extract_object_names(response: Dict[str, Any]) -> List[str]:
    return [
        obj["Key"]
        for obj in response["Contents"]
        if not obj["Key"].endswith("/")
        and obj["Key"] != "metrics.tar.gz"
        and not obj["Key"].endswith("config.json")
        and not obj["Key"].endswith("performance.json")
    ]


def _move_job(job: Job, target: Path, include_forecasts: bool):
    job.save(target, include_forecasts=include_forecasts)
