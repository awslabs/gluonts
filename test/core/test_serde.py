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

from collections import Counter
from functools import partial
from operator import add
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import pytest

from gluonts.core.component import equals, equals_list
from gluonts.core import serde


def test_registry_restricts_and_extends_decode():
    encoded = {
        "__kind__": "instance",
        "class": "collections.Counter",
        "args": [["a", "a"]],
    }

    with pytest.raises(ValueError):
        serde.decode(encoded)

    registry = serde.DEFAULT_DECODE_REGISTRY.copy().register(Counter)
    assert serde.decode(encoded, registry=registry) == {"a": 2}


def test_load_json_unsafe_override():
    encoded = serde.dump_json(
        {
            "__kind__": "instance",
            "class": "collections.Counter",
            "args": [["a", "a"]],
        }
    )
    assert serde.load_json(encoded, unsafe=True) == {"a": 2}


class Span(NamedTuple):
    path: Path
    line: int


class BestEpochInfo(NamedTuple):
    params_path: Path
    epoch_no: int
    metric_value: float


# Example Instances
# -----------------

best_epoch_info = BestEpochInfo(
    params_path=Path("foo/bar"), epoch_no=1, metric_value=0.5
)


numpy_array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)

list_container = [
    best_epoch_info,
    42,
    0.7,
    "fx",
    numpy_array,
]

set_container = {best_epoch_info, 42, 0.7, "fx"}

dict_container = dict(best_epoch_info=best_epoch_info)

simple_types = [
    1,
    42.0,
    "Oh, Romeo",
    np.int32(33),
    np.float64(3.1415),
]  # float('nan')

complex_types = [
    Path("foo/bar"),
    best_epoch_info,
    numpy_array,
]

container_types = [list_container, dict_container, set_container]

examples = simple_types + complex_types + container_types  # type: ignore


def check_equality(expected, actual) -> bool:
    if isinstance(expected, set):
        # Sets are serialized as lists — we check if they have the same elements
        return equals_list(
            sorted(expected, key=hash), sorted(actual, key=hash)
        )
    elif np.issubdtype(type(expected), np.integer):
        # Integer types are expected to be equal exactly
        return np.equal(expected, actual)
    elif np.issubdtype(type(expected), np.inexact):
        # Floating point types are expected to be equal up a certain digit, as specified in np.isclose
        return np.allclose(expected, actual)
    else:
        return equals(expected, actual)


@pytest.mark.parametrize("e", examples)
def test_json_serialization(e) -> None:
    registry = serde.DEFAULT_DECODE_REGISTRY.copy().register(
        BestEpochInfo, Span
    )
    expected, actual = e, serde.load_json(
        serde.dump_json(e), registry=registry
    )
    assert check_equality(expected, actual)


def test_registered_gluonts_base_family():
    from gluonts.time_feature import Constant

    constant = Constant(value=1.0)
    assert serde.decode(serde.encode(constant)) == constant


def test_timestamp_encode_decode() -> None:
    now = pd.Timestamp.now()
    assert now == serde.decode(serde.encode(now))


def test_string_escape() -> None:
    assert serde.load_json(serde.dump_json(r"a\b")) == r"a\b"


def test_serde_fq():
    registry = serde.DEFAULT_DECODE_REGISTRY.copy().register(add)
    add_ = serde.decode(serde.encode(add), registry=registry)
    assert add_(1, 2) == 3

    def foo():
        pass

    with pytest.raises(Exception):
        serde.encode(foo)


def test_serde_partial():
    add_1 = partial(add, 1)

    registry = serde.DEFAULT_DECODE_REGISTRY.copy().register(add)
    add_1_ = serde.decode(serde.encode(add_1), registry=registry)

    assert add_1_(2) == 3


class X(serde.Stateless):
    def m(self):
        return 42


def test_serde_method():
    x = X()

    registry = serde.DEFAULT_DECODE_REGISTRY.copy().register(X)
    m = serde.decode(serde.encode(x.m), registry=registry)

    assert m() == 42


def test_np_str_dtype():
    a = np.array(["foo"])
    serde.decode(serde.encode(a.dtype)) == a.dtype


def test_serde_init_passed_kwargs():
    # classes encoded via `__init_passed_kwargs__` (e.g. zebras periods) must
    # round-trip through the allowlist
    from gluonts import zebras as zb

    period = zb.period("2021-01-01", "D")
    periods = zb.periods("2021-01-01", "D", 10)
    registry = serde.DEFAULT_DECODE_REGISTRY.copy().register(
        type(period), type(periods), type(period.freq)
    )

    for obj in [period, periods]:
        assert serde.decode(serde.encode(obj), registry=registry) == obj


def test_serde_dataclass_instance():
    # dataclasses that set `__init_passed_kwargs__` on the instance (not the
    # class), such as `transform.Chain`, must round-trip through the allowlist
    from gluonts.transform import Chain, Identity

    chain = Chain([Identity(), Identity()])
    registry = serde.DEFAULT_DECODE_REGISTRY.copy().register(Chain, Identity)
    assert serde.decode(serde.encode(chain), registry=registry) == chain


@pytest.mark.parametrize(
    "obj",
    [
        {"__kind__": 42, "class": cls_str}
        for cls_str in [
            "builtins.eval",
            "builtins.exec",
            "builtins.compile",
            "builtins.open",
            "builtins.input",
            "eval",
            "exec",
            "compile",
            "open",
            "input",
        ]
    ],
)
def test_decode_rejects_unregistered_builtins(obj):
    with pytest.raises(ValueError):
        serde.decode(obj)


# `decode` only resolves registered targets; other names must be rejected
# before they can be instantiated or imported.
@pytest.mark.parametrize(
    "class_name",
    [
        "subprocess.run",
        "subprocess.Popen",
        "subprocess.call",
        "os.system",
        "os.popen",
        "ctypes.CDLL",
        "builtins.__import__",
        "webbrowser.open",
    ],
)
@pytest.mark.parametrize("kind", ["type", "instance", "stateful"])
def test_decode_rejects_arbitrary_callables(class_name, kind):
    with pytest.raises(ValueError):
        serde.decode(
            {
                "__kind__": kind,
                "class": class_name,
                "args": [],
                "kwargs": {},
            }
        )


def test_decode_rejects_unknown_class_in_predictor(tmp_path):
    # End-to-end: a predictor.json referencing a class `encode` never produces
    # is rejected when loaded through the deserialize API.
    from gluonts.model.predictor import Predictor

    canary = tmp_path / "canary"
    (tmp_path / "gluonts-config.json").write_text(
        serde.dump_json(
            {
                "type": "gluonts.model.predictor.RepresentablePredictor",
                "version": "test",
            }
        )
    )
    (tmp_path / "predictor.json").write_text(
        serde.dump_json(
            {
                "__kind__": "instance",
                "class": "subprocess.run",
                "args": [["/bin/sh", "-c", f"touch {canary}"]],
                "kwargs": {},
            }
        )
    )

    with pytest.raises(ValueError):
        Predictor.deserialize(tmp_path)

    assert not canary.exists()
