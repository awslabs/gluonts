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

# Standard library imports
import json
import socket
import tempfile
import time
from contextlib import closing, contextmanager
from multiprocessing import Process
from pathlib import Path
from typing import Any, ContextManager, Dict, Optional, Type

# First-party imports
from gluonts.dataset.repository.datasets import materialize_dataset
from gluonts.model.predictor import Predictor
from gluonts.shell.sagemaker import ServeEnv, ServePaths, TrainEnv, TrainPaths
from gluonts.shell.sagemaker.params import encode_sagemaker_parameters
from gluonts.shell.serve import ServerFacade, Settings, make_gunicorn_app


def free_port() -> int:
    """Returns a random unbound port."""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


@contextmanager  # type: ignore
def temporary_server(
    env: Optional[ServeEnv],
    forecaster_type: Optional[Type[Predictor]],
    settings: Settings = Settings(),
) -> ContextManager[ServerFacade]:
    """
    A context manager that instantiates a Gunicorn inference server in a
    separate process (using the :func:`make_inference_server` call)

    Parameters
    ----------
    env
        The :class:`ServeEnv` to use in static inference mode.
        Either `env` or `forecaster_type` must be set.
    forecaster_type
        The :class:`Predictor` type to use in dynamic inference mode.
        Either `env` or `forecaster_type` must be set.
    settings
        Settings to use when instantiating the Gunicorn server.

    Returns
    -------
    ContextManager[ServerFacade]
        A context manager that yields the :class:`InferenceServer` instance
        wrapping the spawned inference server.
    """

    gunicorn_app = make_gunicorn_app(env, forecaster_type, settings)
    process = Process(target=gunicorn_app.run)
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
        The name of the repository dataset to use when instantiating the
        training environment.
    dataset_name
        The name of the repository dataset to use when instantiating the
        training environment.

    Returns
    -------
    ContextManager[TrainEnv]
        A context manager that yields the :class:`TrainEnv` instance.
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
    :class:`Predictor` in a temporary directory and removes the directory on
    exit.

    Parameters
    ----------
    predictor
        A predictor to serialize in :class:`ServeEnv` `model` folder.

    Returns
    -------
    ContextManager[TrainEnv]
        A context manager that yields the :class:`ServeEnv` instance.
    """

    with tempfile.TemporaryDirectory(prefix="gluonts-serve-env") as base:
        paths = ServePaths(base=Path(base))

        # serialize model
        predictor.serialize(paths.model)

        yield ServeEnv(path=paths.base)
