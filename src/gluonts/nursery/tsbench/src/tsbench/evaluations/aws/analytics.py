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
import datetime
import json
import logging
import re
import shutil
import tempfile
import time
from collections import Counter
from functools import cached_property, lru_cache
from pathlib import Path
from typing import Any, Callable, cast, Dict, Iterator, List, Tuple, Union
import boto3
import numpy as np
from botocore.exceptions import ClientError
from .session import default_session


# -------------------------------------------------------------------------------------------------
class Artifact:
    """
    An artifact manages an untarred model artifact of a training job. More
    precisely, it manages a local temporary directory which contains all files
    stored as artifacts.

    The artifact ought to be used within a `with` statement. Upon exit, the temporary directory is
    cleaned up.

    Attributes:
        path: The path of the artifact's managed directory.
    """

    def __init__(self, path: Path, cleanup: bool):
        """
        Initializes a new artifact in the specified directory.

        **Note: Do not call this initializer yourself. It is merely returned when accessing the
        artifacts of a training job.**
        """
        self.path = path
        self.cleanup = cleanup

    def __enter__(self) -> Artifact:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self.cleanup:
            shutil.rmtree(self.path)


# -------------------------------------------------------------------------------------------------
class TrainingJob:
    """
    A training job represents a Sagemaker training job within an experiment.
    """

    def __init__(self, info: Any):
        """
        Initializes a new training job, using the specified boto3 session.

        **Note: This method should only be called in the context of an Analysis object. Do not use
        this initializer yourself.**
        """
        self.info = info

    @property
    def name(self) -> str:
        """
        Returns the name of the training job.
        """
        return self.info["TrainingJobName"]

    @property
    def status(self) -> str:
        """
        Returns the status of the training job.
        """
        return self.info["TrainingJobStatus"]

    @property
    def date_created(self) -> datetime.datetime:
        """
        Returns the date and time when the training job was created.
        """
        return self.info["CreationTime"]

    @property
    def hyperparameters(self) -> dict[str, Any]:
        """
        Returns all user-defined hyper parameters.
        """
        return {
            k: _process_hyperparameter_value(v)
            for k, v in self.info["HyperParameters"].items()
            if not k.startswith("sagemaker_")
            and not k.endswith("_output_distribution")
        }

    @lru_cache()
    def pull_logs(self) -> list[str]:
        """
        Pulls the training job's logs such that subsequent accesses to the
        `logs` property are noops.
        """
        # Check if the logs are already available locally
        log_file = self._cache_dir() / "logs.txt"
        if log_file.exists():
            with log_file.open("r") as f:
                return f.read().split("\n")

        # If not, fetch them
        client = default_session().client("logs")
        streams = client.describe_log_streams(
            logGroupName="/aws/sagemaker/TrainingJobs",
            logStreamNamePrefix=self.info["TrainingJobName"],
        )
        res = []
        for stream in streams["logStreams"]:
            params = {
                "logGroupName": "/aws/sagemaker/TrainingJobs",
                "logStreamName": stream["logStreamName"],
                "startFromHead": True,
            }
            result = client.get_log_events(**params)
            res.extend([event["message"] for event in result["events"]])
            while "nextForwardToken" in result:
                next_token = result["nextForwardToken"]
                result = client.get_log_events(nextToken=next_token, **params)
                if result["nextForwardToken"] == next_token:
                    # The same token as before indicates end of stream, see
                    # https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/logs.html#CloudWatchLogs.Client.get_log_events
                    break
                res.extend([event["message"] for event in result["events"]])

        # Store them
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("w") as f:
            f.write("\n".join(res))

        # And return them
        return res

    @property
    def logs(self) -> list[str]:
        """
        Retrieves the logs emitted by this training job.
        """
        # We can't put the `pull_logs` code here directly since `cached_property` seems to be CPU-
        # bound for some odd reason.
        return self.pull_logs()

    @cached_property
    def metrics(self) -> dict[str, np.ndarray]:
        """
        Fetches the metrics defined by the training script from the training
        job's logs.

        For each metric, it returns a 1D NumPy array (ordered chronologically).
        """
        # Check if the logs are already available locally
        metrics_file = self._cache_dir() / "metrics.json"
        if metrics_file.exists():
            with metrics_file.open("r") as f:
                return {
                    k: np.array(v, dtype=np.float32)
                    for k, v in json.load(f).items()
                }

        # If not, get them from the logs, write them to the file system and return
        metrics = {
            metric["Name"]: [
                float(x)
                for x in re.findall(metric["Regex"], "\n".join(self.logs))
            ]
            for metric in self.info["AlgorithmSpecification"][
                "MetricDefinitions"
            ]
        }
        with metrics_file.open("w+") as f:
            json.dump(metrics, f)

        # Return them as numpy arrays
        return {k: np.array(v, dtype=np.float32) for k, v in metrics.items()}

    def artifact(self, cache: bool = True) -> Artifact:
        """
        Retrieves the model artifact from S3 and stores it locally in a
        temporary directory.

        Args:
            cache: Whether to cache the extracted artifact.

        Returns:
            The artifact which contains the untarred model artifact directory. The artifact should
                be wrapped in a `with` statement such that the directory is cleaned up after usage.
        """
        cache_dir = self._cache_dir() / "artifacts"

        # First, we check whether the model is already available locally. For this, the `cache`
        # flag is irrelevant
        if cache_dir.exists():
            return Artifact(cache_dir, cleanup=False)

        # If not, we need to download the artifact. For that, we need to get the bucket and object
        # path
        regex = r"^s3://([A-z0-9-_]*)/(.*)$"
        bucket_name, object_path = re.findall(
            regex, self.info["ModelArtifacts"]["S3ModelArtifacts"]
        )[0]

        # Then, we can download the model
        s3 = default_session().client("s3")
        with tempfile.NamedTemporaryFile(suffix=".tar.gz") as tmp:
            s3.download_fileobj(bucket_name, object_path, tmp)
            tmp.seek(0)
            # As soon as it is downloaded, we can unpack the tar into the cache directory or a
            # temporary one
            if cache:
                cache_dir.mkdir(exist_ok=True, parents=True)
                target = cache_dir
            else:
                target = Path(tempfile.mkdtemp())
            shutil.unpack_archive(tmp.name, target)

        # And return the artifact
        return Artifact(target, cleanup=not cache)

    def move_to(self, experiment: str) -> None:
        """
        Updates the experiment tag to the provided name.
        """
        client = default_session().client("sagemaker")
        client.add_tags(
            ResourceArn=self.info["TrainingJobArn"],
            Tags=[{"Key": "Experiment", "Value": experiment}],
        )

    def delete(self) -> None:
        """
        Deletes the training job by removing all tags associated with it.
        """
        client = default_session().client("sagemaker")

        existing_tags = client.list_tags(
            ResourceArn=self.info["TrainingJobArn"],
            MaxResults=100,
        )
        experiment = [
            t["Value"]
            for t in existing_tags["Tags"]
            if t["Key"] == "Experiment"
        ][0]

        client.add_tags(
            ResourceArn=self.info["TrainingJobArn"],
            Tags=[{"Key": "OriginalExperiment", "Value": experiment}],
        )

        client.delete_tags(
            ResourceArn=self.info["TrainingJobArn"],
            TagKeys=["Experiment"],
        )

    def __repr__(self) -> str:
        return f"TrainingJob(name={self.info['TrainingJobName']})"

    def _cache_dir(self) -> Path:
        return (
            Path.home()
            / "tsbench"
            / "cache"
            / cast(str, self.info["TrainingJobName"])
        )


# -------------------------------------------------------------------------------------------------
class Analysis:
    """
    The analysis object allows analyzing a set of training jobs that belong to
    the same experiment.
    """

    def __init__(
        self,
        experiment: str,
        only_completed: bool = True,
        include: Callable[[TrainingJob], bool] = lambda _: True,
        resolve_duplicates: bool = True,
    ):
        """
        Initializes a new analysis object, using the specified session to make
        requests to AWS and Sagemaker. The initializer already fetches all
        training jobs belonging to the provided experiment.

        Args:
            session: The session to interact with AWS services.
            experiment: The name of the experiment to analyze.
            only_completed: Whether to ignore runs which have not completed successfully (a
                warning will be emitted nonetheless).
            include: Whether the training job should be included in the summary. By default, it
                returns True for any job. If `only_completed` is set to True, only completed jobs
                will be passed to this callback.
            resolve_duplicates: Whether to exclude the older experiments if experiments with the
                same hyperparameters are found.
        """
        self.experiment_name = experiment
        training_jobs, duplicates = _fetch_training_jobs(
            default_session(),
            self.experiment_name,
            only_completed,
            resolve_duplicates,
        )
        self.duplicates = duplicates
        self.map = {t.name: t for t in training_jobs if include(t)}
        if len(self.map) < len(training_jobs):
            logging.warning(
                " Analysis manually excludes %d jobs",
                len(training_jobs) - len(self.map),
            )

    def get(self, name: str) -> TrainingJob:
        """
        Returns the training job with the specified name.
        """
        return self.map[name]

    @property
    def status(self) -> dict[str, int]:
        """
        Returns the aggregate statistics about the status of all jobs.
        """
        c = Counter([t.status for t in self.map.values()])
        return dict(c)

    def __iter__(self) -> Iterator[TrainingJob]:
        return iter(self.map.values())

    def __len__(self) -> int:
        return len(self.map)

    def __repr__(self) -> str:
        return (
            f"Analysis(experiment='{self.experiment_name}',"
            f" num_jobs={len(self):,})"
        )


# -------------------------------------------------------------------------------------------------
def _fetch_training_jobs(
    session: boto3.Session,
    experiment: str,
    only_completed: bool,
    resolve_duplicates: bool,
) -> tuple[list[TrainingJob], list[TrainingJob]]:
    """
    Fetches all training jobs which are associated with this experiment.
    """
    client = session.client("sagemaker")
    search_params = {
        "MaxResults": 100,
        "Resource": "TrainingJob",
        "SearchExpression": {
            "Filters": [
                {
                    "Name": "Tags.Experiment",
                    "Operator": "Equals",
                    "Value": experiment,
                }
            ],
        },
    }

    while True:
        try:
            response = client.search(**search_params)
            break
        except ClientError:
            time.sleep(1)

    results = response["Results"]
    while "NextToken" in response:
        while True:
            try:
                response = client.search(
                    NextToken=response["NextToken"], **search_params
                )
                results.extend(response["Results"])
                break
            except ClientError:
                time.sleep(1)

    jobs = [TrainingJob(r["TrainingJob"]) for r in results]

    if only_completed:
        completed_jobs = [j for j in jobs if j.status == "Completed"]
        if len(completed_jobs) < len(jobs):
            c = Counter([j.status for j in jobs])
            d = dict(c)
            del d["Completed"]
            logging.warning(
                " Analysis is ignoring %d jobs %s",
                len(jobs) - len(completed_jobs),
                d,
            )
        jobs = completed_jobs

    duplicates = []
    if resolve_duplicates:
        unique = {}
        for job in jobs:
            hyperparameters = frozenset(job.hyperparameters.items())
            if hyperparameters in unique:
                # Replace existing job if this one is newer. Don't do anything otherwise.
                if unique[hyperparameters].date_created < job.date_created:
                    duplicates.append(unique[hyperparameters])
                    unique[hyperparameters] = job
                else:
                    duplicates.append(job)
            else:
                unique[hyperparameters] = job

        if len(unique) < len(jobs):
            logging.warning(
                " Analysis is ignoring %d superseded jobs",
                len(jobs) - len(unique),
            )
        jobs = list(unique.values())

    return jobs, duplicates


# -------------------------------------------------------------------------------------------------
def _process_hyperparameter_value(v: str) -> str | float | int | bool:
    if re.match(r'^"(.*)"$', v):  # value is a string
        return v[1:-1]
    if v in ("false", "False", "true", "True"):
        return v in ("true", "True")
    if "." in v:  # value is float
        return float(v)
    return int(v)  # value is int
