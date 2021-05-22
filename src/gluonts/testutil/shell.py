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

import json
import multiprocessing
import socket
import tempfile
import time
import typing
from contextlib import closing, contextmanager
from multiprocessing.context import ForkContext
from pathlib import Path
from typing import Any, ContextManager, Dict, Iterable, List, Optional, Type

import requests
from gluonts.dataset.common import DataEntry, serialize_data_entry
from gluonts.dataset.repository.datasets import materialize_dataset
from gluonts.model.predictor import Predictor
from gluonts.shell.env import ServeEnv, TrainEnv
from gluonts.shell.sagemaker import ServePaths, TrainPaths
from gluonts.shell.sagemaker.params import encode_sagemaker_parameters
from gluonts.shell.serve import Settings, make_gunicorn_app


class ServerFacade:
    """
    A convenience wrapper for sending requests and handling responses to
    an inference server located at the given address.
    """

    def __init__(self, base_address: str) -> None:
        self.base_address = base_address

    def url(self, path) -> str:
        return self.base_address + path

    def ping(self) -> bool:
        try:
            response = requests.get(url=self.url("/ping"))
            return response.status_code == 200
        except requests.exceptions.ConnectionError:
            return False

    def execution_parameters(self) -> dict:
        response = requests.get(
            url=self.url("/execution-parameters"),
            headers={"Accept": "application/json"},
        )

        if response.status_code == 200:
            return response.json()
        elif response.status_code >= 400:
            raise RuntimeError(response.content.decode("utf-8"))
        else:
            raise RuntimeError(f"Unexpected {response.status_code} response")

    def invocations(
        self, data_entries: Iterable[DataEntry], configuration: dict
    ) -> List[dict]:
        instances = list(map(serialize_data_entry, data_entries))
        response = requests.post(
            url=self.url("/invocations"),
            json={"instances": instances, "configuration": configuration},
            headers={"Accept": "application/json"},
        )

        if response.status_code == 200:
            predictions = response.json()["predictions"]
            assert len(predictions) == len(instances)
            return predictions
        elif response.status_code >= 400:
            raise RuntimeError(response.content.decode("utf-8"))
        else:
            raise RuntimeError(f"Unexpected {response.status_code} response")

    def batch_invocations(
        self, data_entries: Iterable[DataEntry]
    ) -> List[dict]:
        instances_pre = map(serialize_data_entry, data_entries)
        instances = list(map(json.dumps, instances_pre))

        response = requests.post(
            url=self.url("/invocations"), data="\n".join(instances)
        )

        if response.status_code != 200:
            raise RuntimeError(response.content.decode("utf-8"))

        predictions = list(map(json.loads, response.text.splitlines()))
        assert len(predictions) == len(instances)
        return predictions


def free_port() -> int:
    """Returns a random unbound port."""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


@contextmanager  # type: ignore
def temporary_server(
    env: ServeEnv,
    forecaster_type: Optional[Type[Predictor]],
    settings: Settings = Settings(),
) -> ContextManager[ServerFacade]:
    """
    A context manager that instantiates a Gunicorn inference server in a
    separate process (using the :func:`make_inference_server` call)

    Parameters
    ----------
    env
        The `ServeEnv` to use in static inference mode.
        Either `env` or `forecaster_type` must be set.
    forecaster_type
        The `Predictor` type to use in dynamic inference mode.
        Either `env` or `forecaster_type` must be set.
    settings
        Settings to use when instantiating the Gunicorn server.

    Returns
    -------
    ContextManager[ServerFacade]
        A context manager that yields the `InferenceServer` instance
        wrapping the spawned inference server.
    """
    context = multiprocessing.get_context("fork")
    context = typing.cast(ForkContext, context)  # cast to make mypi pass

    gunicorn_app = make_gunicorn_app(env, forecaster_type, settings)
    process = context.Process(target=gunicorn_app.run)
    process.start()

    endpoint = ServerFacade(
        base_address="http://{address}:{port}".format(
            address=settings.sagemaker_server_address,
            port=settings.sagemaker_server_port,
        )
    )

    # try to ping the server (signalling liveness)
    # poll for n seconds in t second intervals
    n, t = 10, 2
    max_time = time.time() + n
    while not endpoint.ping():
        if time.time() < max_time:
            time.sleep(t)
        else:
            msg = f"Failed to start the inference server within {n} seconds"
            raise TimeoutError(msg)

    yield endpoint

    process.terminate()
    process.join()


@contextmanager  # type: ignore
def temporary_train_env(
    hyperparameters: Dict[str, Any], dataset_name: str
) -> ContextManager[TrainEnv]:
    """
    A context manager that instantiates a training environment from a given
    combination of `hyperparameters` and `dataset_name` in a temporary
    directory and removes the directory on exit.

    Parameters
    ----------
    hyperparameters
        The hyperparameters to use when instantiating the
        training environment.
    dataset_name
        The name of the repository dataset to use when instantiating the
        training environment.

    Returns
    -------
    ContextManager[gluonts.shell.env.TrainEnv]
        A context manager that yields the `TrainEnv` instance.
    """

    with tempfile.TemporaryDirectory(prefix="gluonts-train-env") as base:
        paths = TrainPaths(base=Path(base))

        # write hyperparameters
        with paths.hyperparameters.open(mode="w") as fp:
            hps_encoded = encode_sagemaker_parameters(hyperparameters)
            json.dump(hps_encoded, fp, indent=2, sort_keys=True)

        # save dataset
        ds_path = materialize_dataset(dataset_name)

        path_metadata = paths.data / "metadata" / "metadata.json"
        path_train = paths.data / "train"
        path_test = paths.data / "test"

        path_metadata.parent.mkdir(exist_ok=True)

        path_metadata.symlink_to(ds_path / "metadata.json")
        path_train.symlink_to(ds_path / "train", target_is_directory=True)
        path_test.symlink_to(ds_path / "test", target_is_directory=True)

        yield TrainEnv(path=paths.base)


@contextmanager  # type: ignore
def temporary_serve_env(predictor: Predictor) -> ContextManager[ServeEnv]:
    """
    A context manager that instantiates a serve environment for a given
    `Predictor` in a temporary directory and removes the directory on
    exit.

    Parameters
    ----------
    predictor
        A predictor to serialize in `ServeEnv` `model` folder.

    Returns
    -------
    ContextManager[gluonts.shell.env.ServeEnv]
        A context manager that yields the `ServeEnv` instance.
    """

    with tempfile.TemporaryDirectory(prefix="gluonts-serve-env") as base:
        paths = ServePaths(base=Path(base))

        # serialize model
        predictor.serialize(paths.model)

        yield ServeEnv(path=paths.base)
