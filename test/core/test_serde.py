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
from pathlib import Path
from typing import List, NamedTuple

# Third-party imports
import mxnet as mx
import numpy as np
import pytest
from pydantic import BaseModel

# First-party imports
from gluonts.core import serde

# Example Types
# -------------
from gluonts.core.component import equals


class Span(NamedTuple):
    path: Path
    line: int


class BestEpochInfo(NamedTuple):
    params_path: Path
    epoch_no: int
    metric_value: float


class CategoricalFeatureInfo(BaseModel):
    name: str
    cardinality: int


class MyGluonBlock(mx.gluon.HybridBlock):
    def __init__(
        self,
        feature_infos: List[CategoricalFeatureInfo],
        feature_dims: List[int],
    ) -> None:
        super().__init__()
        self.feature_infos = feature_infos
        self.feature_dims = feature_dims

    def hybrid_forward(self, F, x, *args, **kwargs):
        raise NotImplementedError

    # required for all user-defined types
    def __getnewargs_ex__(self):
        return (self.feature_infos, self.feature_dims), dict()

    def __eq__(self, that) -> bool:
        if isinstance(that, MyGluonBlock):
            return self.__getnewargs_ex__() == that.__getnewargs_ex__()
        else:
            return False


# Example Instances
# -----------------

best_epoch_info = BestEpochInfo(
    params_path=Path("foo/bar"), epoch_no=1, metric_value=0.5
)

feature_info = CategoricalFeatureInfo(name="cat", cardinality=10)

custom_type = MyGluonBlock(feature_infos=[feature_info], feature_dims=[10])

numpy_array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)

list_container = [
    best_epoch_info,
    feature_info,
    custom_type,
    42,
    0.7,
    "fx",
    numpy_array,
]

dict_container = dict(
    best_epoch_info=best_epoch_info,
    feature_info=feature_info,
    custom_type=custom_type,
)

simple_types = [1, 42.0, "Oh, Romeo"]  # float('nan')

complex_types = [
    Path("foo/bar"),
    best_epoch_info,
    feature_info,
    custom_type,
    numpy_array,
]

container_types = [list_container, dict_container]

examples = simple_types + complex_types + container_types  # type: ignore


@pytest.mark.parametrize("e", examples)
def test_binary_serialization(e) -> None:
    assert equals(e, serde.load_binary(serde.dump_binary(e)))


@pytest.mark.parametrize("e", examples)
def test_json_serialization(e) -> None:
    assert equals(e, serde.load_json(serde.dump_json(e)))


@pytest.mark.parametrize("e", examples)
def test_code_serialization(e) -> None:
    assert equals(e, serde.load_code(serde.dump_code(e)))
