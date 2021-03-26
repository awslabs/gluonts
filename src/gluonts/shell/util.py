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

import pydoc
from typing import Type, Union, cast

import pkg_resources

from gluonts.core.exception import GluonTSForecasterNotFoundError
from gluonts.model.estimator import Estimator
from gluonts.model.predictor import Predictor

Forecaster = Type[Union[Estimator, Predictor]]


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
