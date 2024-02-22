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

from gluonts.shell.sagemaker.nested_params import decode_nested_parameters


def test_nested_params():
    data = decode_nested_parameters(
        {
            "$env.num_workers": "4",
            "$evaluation.quantiles": [0.1, 0.5, 0.9],
            "prediction_length": 14,
        }
    )

    hps = data.pop("")
    assert hps["prediction_length"] == 14

    env = data.pop("env")
    assert env["num_workers"] == "4"

    evaluation = data.pop("evaluation")
    assert evaluation["quantiles"] == [0.1, 0.5, 0.9]
