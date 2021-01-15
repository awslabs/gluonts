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

from typing import TypeVar, Dict, Any, List, Tuple
import json

from ._base import encode, decode

T = TypeVar("T", bound="Copyable")  # type: ignore


def clone_with_params(t: T, params: Dict[str, Any]) -> T:
    """
    Return a copy of the object with some parameters overwritten.
    Use dots for nested parameters.

    e.g.

        clone_with_params(my_estimator, {"trainer.epochs": 2})

    """
    json_blob = encode(t)

    def update(js, key, value):
        key_parts = key.split(".")
        if "kwargs" in js:
            update(js["kwargs"], key, value)
        elif len(key_parts) == 1:
            assert key in js
            js[key] = value
        else:
            update(js[key_parts[0]], ".".join(key_parts[1:]), value)

    for k, v in params.items():
        update(json_blob["kwargs"], k, encode(v))
    return decode(json_blob)


def get_flat_params(t: Any, prefix="") -> List[Tuple[str, str]]:
    """
    Returns a flat list of all the parameters of the model estimator.
    The main use case for this is allow easy tracking/logging of all parameters and settings during an experiment.

    E.g.
    get_flat_params(my_estimator, prefix="estimator")
    [
        ('estimator', 'gluonts.model.deepar._estimator.DeepAREstimator'),
        ('estimator.cell_type', 'lstm'),
        ('estimator.distr_output', 'gluonts.mx.distribution.student_t.StudentTOutput'),
        ('estimator.trainer', 'gluonts.mx.trainer._base.Trainer'),
        ('estimator.trainer.avg_strategy', 'gluonts.mx.trainer.model_averaging.SelectNBestMean'),
        ('estimator.trainer.avg_strategy.maximize', 'False'),
        ('estimator.trainer.avg_strategy.metric', 'score'),
        ('estimator.trainer.avg_strategy.num_models', 1)
    ]
    """
    json_blob = encode(t)
    flat = []

    def iter_json(js, path):
        if isinstance(js, dict) and js.get("__kind__") == "instance":
            flat.append((path, js["class"]))
            for k in js["kwargs"].keys():
                iter_json(js["kwargs"][k], f"{path}.{k}")
        else:
            flat.append((path, js))

    iter_json(json_blob, prefix)
    return flat
