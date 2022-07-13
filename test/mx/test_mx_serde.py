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

from textwrap import dedent
from typing import List

import pytest

import mxnet as mx
import numpy as np
from pydantic import BaseModel

from gluonts.core import serde
from gluonts.core.component import equals

import gluonts.mx.prelude as _


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


feature_info = CategoricalFeatureInfo(name="cat", cardinality=10)
custom_type = MyGluonBlock(feature_infos=[feature_info], feature_dims=[10])

examples = [feature_info, custom_type]


@pytest.mark.parametrize("e", examples)
def test_json_serialization(e) -> None:
    expected, actual = e, serde.load_json(serde.dump_json(e))
    assert equals(expected, actual)


@pytest.mark.parametrize("e", examples)
def test_code_serialization(e) -> None:
    expected, actual = e, serde.load_code(serde.dump_code(e))
    assert equals(expected, actual)


@pytest.mark.parametrize(
    "a",
    [
        mx.nd.random.uniform(shape=(3, 5, 2), dtype="float16"),
        mx.nd.random.uniform(shape=(3, 5, 2), dtype="float32"),
        mx.nd.random.uniform(shape=(3, 5, 2), dtype="float64"),
        mx.nd.array([[1, 2, 3], [-1, -2, 0]], dtype=np.uint8),
        mx.nd.array([[1, 2, 3], [-1, -2, 0]], dtype=np.int32),
        mx.nd.array([[1, 2, 3], [-1, -2, 0]], dtype=np.int64),
        mx.nd.array([[1, 2, 3], [1, 2, 0]], dtype=np.uint8),
    ],
)
@pytest.mark.parametrize(
    "serialize_fn",
    [
        lambda x: serde.load_json(serde.dump_json(x)),
        lambda x: serde.load_code(serde.dump_code(x)),
    ],
)
def test_ndarray_serialization(a, serialize_fn) -> None:
    b = serialize_fn(a)
    assert type(a) == type(b)
    assert a.dtype == b.dtype
    assert a.shape == b.shape
    assert np.all((a == b).asnumpy())


def test_dynamic_loading():
    code = dedent(
        """
        dict(
           trainer=gluonts.mx.trainer.Trainer(
               ctx="cpu(0)",
               epochs=5,
               learning_rate=0.001,
               clip_gradient=10.0,
               weight_decay=1e-08,
               num_batches_per_epoch=10,
               hybridize=False,
           ),
           num_hidden_dimensions=[3],
           context_length=5,
           prediction_length=2,
           freq="1H",
           distr_output=gluonts.mx.distribution.StudentTOutput(),
           batch_normalization=False,
           mean_scaling=True
        )
        """
    )

    serde.load_code(code)
