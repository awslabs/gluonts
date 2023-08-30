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
import time
from datetime import datetime, timezone
from pathlib import Path
import click
import sagemaker
from botocore.exceptions import ClientError
from cli.utils import generate_configurations, iterate_configurations
from tsbench.evaluations.aws import default_session
from tsbench.evaluations.aws.ecr import image_uri
from tsbench.evaluations.aws.framework import CustomFramework
from tsbench.evaluations.metrics.sagemaker import metric_definitions
from ._main import evaluations


@evaluations.command(short_help="Schedule evaluations on AWS Sagemaker.")
@click.option(
    "--config_path",
    required=True,
    help=(
        "The local path to the configuration file or directory. For"
        " directories, all YAML files in the directory and its subfolders are"
        " considered."
    ),
)
@click.option(
    "--sagemaker_role",
    required=True,
    help=(
        "The IAM role to use for the AWS Sagemaker instance and for"
        " downloading data locally. This should be the full ARN."
    ),
)
@click.option(
    "--experiment",
    default="tsbench",
    show_default=True,
    help=(
        "The name of the experiment which allows grouping training runs on AWS"
        " Sagemaker."
    ),
)
@click.option(
    "--data_bucket",
    default="tsbench",
    show_default=True,
    help="The S3 bucket where the datasets have been uploaded to.",
)
@click.option(
    "--data_bucket_prefix",
    default="data",
    show_default=True,
    help="The prefix for the S3 bucket where datasets have been uploaded to.",
)
@click.option(
    "--output_bucket",
    default="tsbench",
    show_default=True,
    help=(
        "The S3 bucket where outputs (model parameters and forecasts) should"
        " be written to."
    ),
)
@click.option(
    "--output_bucket_prefix",
    default="evaluations",
    show_default=True,
    help="The prefix for the S3 bucket where outputs are written to.",
)
@click.option(
    "--instance_type",
    default="ml.c5.2xlarge",
    show_default=True,
    help=(
        "The type of instance to use for running evaluations if not provided"
        " explicitly."
    ),
)
@click.option(
    "--docker_image",
    default="tsbench:latest",
    show_default=True,
    help=(
        "The Docker image in your ECR registry to use for running the"
        " evaluation."
    ),
)
@click.option(
    "--max_runtime",
    default=240,
    show_default=True,
    help=(
        "The maximum number of hours for which individual evaluations may run."
    ),
)
@click.option(
    "--nskip",
    default=0,
    show_default=True,
    help=(
        "The number of configurations to skip. Useful if some set of"
        " evaluations failed."
    ),
)
@click.option(
    "--local",
    default=False,
    show_default=True,
    help=(
        "Whether to run evaluations locally via Docker Compose instead of on"
        " AWS Sagemaker."
    ),
)
def schedule(
    config_path: str,
    sagemaker_role: str,
    experiment: str,
    data_bucket: str,
    data_bucket_prefix: str,
    output_bucket: str,
    output_bucket_prefix: str,
    instance_type: str,
    docker_image: str,
    max_runtime: int,
    nskip: int,
    local: bool,
):
    """
    Schedules evaluations on AWS Sagemaker by running a grid search over the
    configurations provided in the given file(s).

    As AWS Sagemaker does not allow queuing jobs, this script is running as
    long as not all evaluation runs have been scheduled.
    """
    assert instance_type[:5] not in (
        "ml.p3",
        "ml.p2",
        "ml.g4",
    ), "Cannot schedule experiments on GPU instances."

    # First, setup Sagemaker connection
    boto_session = default_session()
    if local:
        sm_session = sagemaker.LocalSession(boto_session)
    else:
        sm_session = sagemaker.Session(boto_session)

    def job_factory() -> str:
        date_str = datetime.now(tz=timezone.utc).strftime(
            "%d-%m-%Y-%H-%M-%S-%f"
        )
        job_name = f"{experiment}-{date_str}"
        return job_name

    # Then, generate configs
    all_configurations = generate_configurations(Path(config_path))

    # Then, we can run the training, passing parameters as required
    for configuration in iterate_configurations(all_configurations, nskip):
        # Create the estimator
        estimator = CustomFramework(
            sagemaker_session=sm_session,
            role=sagemaker_role,
            tags=[
                {"Key": "Experiment", "Value": experiment},
            ],
            instance_type="local"
            if local
            else (
                configuration["__instance_type__"]
                if "__instance_type__" in configuration
                else instance_type
            ),
            instance_count=1,
            volume_size=30,
            max_run=max_runtime * 60 * 60,
            image_uri=image_uri(docker_image),
            source_dir=str(
                Path(os.path.realpath(__file__)).parent.parent.parent
            ),
            output_path=(
                f"s3://{output_bucket}/{output_bucket_prefix}/{experiment}"
            ),
            entry_point="evaluate.py",
            debugger_hook_config=False,
            metric_definitions=metric_definitions(),
            hyperparameters={
                k: v
                for k, v in configuration.items()
                if not k.startswith("__")
            },
        )

        while True:
            # Try fitting the estimator
            try:
                estimator.fit(
                    job_name=job_factory(),
                    inputs={
                        configuration[
                            "dataset"
                        ]: f"s3://{data_bucket}/{data_bucket_prefix}/{configuration['dataset']}"
                    },
                    wait=False,
                )
                break
            except ClientError as err:
                print(f"+++ Scheduling failed: {err}")
                print("+++ Sleeping for 5 minutes.")
                time.sleep(300)

        print(f">>> Launched job: {estimator.latest_training_job.name}")  # type: ignore

    print(">>> Successfully scheduled all training jobs.")
