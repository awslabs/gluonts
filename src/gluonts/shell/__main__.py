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
import logging
import pydoc
import traceback
from pathlib import Path
from typing import Optional, Type, Union, cast

# Third-party imports
import click
import pkg_resources

# First-party imports
from gluonts.core.exception import GluonTSForecasterNotFoundError
from gluonts.model.estimator import Estimator
from gluonts.model.predictor import Predictor

# Relative imports
from gluonts.shell.serve import Settings
from .sagemaker import TrainEnv, ServeEnv

Forecaster = Type[Union[Estimator, Predictor]]

logger = logging.getLogger(__name__)


def forecaster_type_by_name(name: str) -> Forecaster:
    """
    Loads a forecaster from the `gluonts_forecasters` entry_points namespace
    by name.

    If a forecater wasn't register under that name, it tries to locate the
    class.

    Third-party libraries can register their forecasters as follows by defining
    a corresponding section in the `entry_points` section of their `setup.py`::

        entry_points={
            'gluonts_forecasters': [
                'model_a = my_models.model_a:MyEstimator',
                'model_b = my_models.model_b:MyPredictor',
            ]
        }
    """
    forecaster = None

    for entry_point in pkg_resources.iter_entry_points("gluonts_forecasters"):
        if entry_point.name == name:
            forecaster = entry_point.load()
            break
    else:
        forecaster = pydoc.locate(name)

    if forecaster is None:
        raise GluonTSForecasterNotFoundError(
            f'Cannot locate estimator with classname "{name}".'
        )

    return cast(Forecaster, forecaster)


@click.group()
def cli() -> None:
    pass


@cli.command(name="serve")
@click.option(
    "--data-path",
    type=click.Path(),
    envvar="SAGEMAKER_DATA_PATH",
    default="/opt/ml",
    help="The root path of all folders mounted by the SageMaker runtime.",
)
@click.option(
    "--forecaster",
    metavar="NAME",
    envvar="GLUONTS_FORECASTER",
    help=(
        "An alias or a fully qualified name of a Predictor to use. "
        "If this value is defined, the inference server runs in the "
        "so-called dynamic mode, where the predictor is initialized for "
        "each request using parameters provided in the 'configuration' field "
        "of the JSON request. Otherwise, the server runs in static mode, "
        "where the predictor is initialized upfront from a serialized model "
        "located in the {data-path}/model folder."
    ),
)
@click.option(
    "--force-static/--no-force-static",
    envvar="GLUONTS_FORCE_STATIC",
    default=False,
    help=(
        "Forces execution in static mode, even in situations where the "
        '"forecaster" option is present.'
    ),
)
def serve_command(
    data_path: str, forecaster: Optional[str], force_static: bool
) -> None:
    from gluonts.shell import serve

    logger.info("Run 'serve' command")

    if not force_static and forecaster is not None:
        forecaster_type: Optional[Forecaster] = forecaster_type_by_name(
            forecaster
        )
    else:
        forecaster_type = None

    gunicorn_app = serve.make_gunicorn_app(
        env=ServeEnv(Path(data_path)),
        forecaster_type=forecaster_type,
        settings=Settings(),
    )
    gunicorn_app.run()


@cli.command(name="train")
@click.option(
    "--data-path",
    type=click.Path(exists=True),
    envvar="SAGEMAKER_DATA_PATH",
    default="/opt/ml",
    help="The root path of all folders mounted by the SageMaker runtime.",
)
@click.option(
    "--forecaster",
    type=str,
    envvar="GLUONTS_FORECASTER",
    help=(
        "An alias or a fully qualified name of a Predictor or Estimator to "
        "use. If this value is not defined, the command will try to read it"
        "from the hyperparameters dictionary under the 'forecaster_name' key. "
        "If the value denotes a Predictor, training will be skipped and the "
        "command will only do an evaluation on the provided test dataset."
    ),
)
def train_command(data_path: str, forecaster: Optional[str]) -> None:
    from gluonts.shell import train

    logger.info("Run 'train' command")
    env = TrainEnv(Path(data_path))

    try:
        if forecaster is None:
            try:
                forecaster = env.hyperparameters["forecaster_name"]
            except KeyError:
                msg = (
                    "Forecaster shell parameter is `None`, but "
                    "the `forecaster_name` key is not defined in the "
                    "hyperparameters.json dictionary."
                )
                raise GluonTSForecasterNotFoundError(msg)

        assert forecaster is not None
        train.run_train_and_test(env, forecaster_type_by_name(forecaster))
    except Exception as error:
        with open(env.path.output / "failure", "w") as out_file:
            out_file.write(str(error))
            out_file.write("\n\n")
            out_file.write(traceback.format_exc())
        raise


if __name__ == "__main__":
    import logging
    import os

    from gluonts import gluonts_tqdm

    if "TRAINING_JOB_NAME" in os.environ:
        gluonts_tqdm.USE_TQDM = False

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
        datefmt="[%Y-%m-%d %H:%M:%S]",
    )
    cli(prog_name=__package__)
