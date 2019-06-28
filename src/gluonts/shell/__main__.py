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

import click


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "--data-path",
    type=click.Path(exists=True),
    envvar="SAGEMAKER_DATA_PATH",
    required=True,
)
def serve(data_path):
    from gluonts.shell.serve import DefaultShell
    DefaultShell(data_path).run()


FORECASTER_BY_NAME = {
    'deepar': 'gluonts.model.deepar.DeepAREstimator',
    'r': 'gluonts.model.r_forecast.RForecastPredictor',
}


@cli.command()
@click.option(
    "--data-path",
    type=click.Path(exists=True),
    envvar="SAGEMAKER_DATA_PATH",
    required=True,
)
@click.argument("forecaster", required=True, envvar="GLUONTS_FORECASTER")
def train(data_path, forecaster):
    if forecaster in FORECASTER_BY_NAME:
        forecaster = FORECASTER_BY_NAME[forecaster]

    from gluonts.shell.train import train

    train(data_path, forecaster)


if __name__ == "__main__":
    cli()
