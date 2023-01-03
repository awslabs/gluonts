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

from typing import List

from pydantic import BaseModel

from gluonts.core import serde


@serde.dataclass
class Estimator:
    prediction_length: int
    context_length: int = serde.OrElse(
        lambda prediction_length: prediction_length * 2
    )

    use_feat_static_cat: bool = True
    cardinality: List[int] = serde.EVENTUAL

    def __eventually__(self, cardinality):
        if not self.use_feat_static_cat:
            cardinality.set([1])
        else:
            cardinality.set_default([1])


class A(BaseModel):
    pass


class B(A):
    pass


@serde.dataclass
class X:
    a: A


def test_dataclass():
    est = Estimator(prediction_length=7)

    assert est.prediction_length == 7
    assert est.context_length == 14
    assert est.cardinality == [1]


def test_dataclass_subtype():
    x = X(a=B())

    assert isinstance(x.a, B)

    x2 = serde.decode(serde.encode(x))
    assert isinstance(x2.a, B)


def test_dataclass_inheritance():
    @serde.dataclass
    class A:
        x: int = 1
        y: int = 2

    @serde.dataclass
    class B(A):
        z: int = 4

    b = B(x=3)
    assert b.x == 3
    assert b.z == 4


def test_dataclass_eventual():
    @serde.dataclass
    class X:
        y: int = serde.EVENTUAL

        def __eventually__(self, y):
            y.set_default(3)

    x1 = X(y=1)
    assert x1.y == 1

    x2 = X()
    assert x2.y == 3
