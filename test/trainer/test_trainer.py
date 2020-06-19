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
from typing import Any, List

# Third-party imports
import pytest

# First-party imports
from gluonts.mx.trainer import Trainer


def test_epochs() -> None:
    assert_valid_param(
        param_name="epochs", param_values=[0, 1, 42, 1000, 1000]
    )
    assert_invalid_param(
        param_name="epochs",
        param_values=[-2, -1],
        exp_msg="The value of `epochs` should be > 0 (type=value_error)",
    )


def test_patience() -> None:
    assert_valid_param(param_name="patience", param_values=[0, 1, 10, 100])
    assert_invalid_param(
        param_name="patience",
        param_values=[-2, -1],
        exp_msg="The value of `patience` should be >= 0 (type=value_error)",
    )


def test_learning_rate() -> None:
    assert_valid_param(
        param_name="learning_rate", param_values=[0.42, 17.8, 10.0]
    )
    assert_invalid_param(
        param_name="learning_rate",
        param_values=[-2, -1e-10, 0, float("inf"), float("nan")],
        exp_msg="The value of `learning_rate` should be > 0 (type=value_error)",
    )


def test_learning_rate_decay_factor() -> None:
    assert_valid_param(
        param_name="learning_rate_decay_factor",
        param_values=[0, 1e-10, 0.5, 1 - 1e-10],
    )
    assert_invalid_param(
        param_name="learning_rate_decay_factor",
        param_values=[-2, -1e-10, +1, +5, float("inf"), float("nan")],
        exp_msg="The value of `learning_rate_decay_factor` should be in the [0, 1) range (type=value_error)",
    )


def assert_valid_param(param_name: str, param_values: List[Any]) -> None:
    try:
        for x in param_values:
            Trainer(**{param_name: x})
    except Exception as e:
        pytest.fail(f'Unexpected exception when initializing Trainer: "{e}"')
        raise e


def assert_invalid_param(
    param_name: str, param_values: List[Any], exp_msg: str
) -> None:
    for x in param_values:
        with pytest.raises(AssertionError) as excinfo:
            Trainer(**{param_name: x})
            assert exp_msg in str(excinfo.value)
