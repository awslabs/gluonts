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

import numpy as np
import pandas as pd
import pytest

import gluonts.dataset.artificial.recipe as rcp

from gluonts.core.component import validated
from gluonts.core.serde import dump_code, load_code
from gluonts.dataset.artificial import RecipeDataset
from gluonts.dataset.artificial.recipe import (
    Add,
    BinaryMarkovChain,
    Constant,
    ConstantVec,
    Debug,
    Env,
    Eval,
    ForEachCat,
    Lag,
    Lifted,
    LinearTrend,
    Mul,
    NanWhere,
    RandomBinary,
    RandomCat,
    RandomGaussian,
    RandomSymmetricDirichlet,
    Ref,
    SmoothSeasonality,
    Stack,
    evaluate,
    generate,
)
from gluonts.dataset.artificial.recipe import lifted_numpy as lnp
from gluonts.dataset.artificial.recipe import take_as_list
from gluonts.dataset.common import (
    BasicFeatureInfo,
    CategoricalFeatureInfo,
    MetaData,
)

BASE_RECIPE = [("foo", ConstantVec(1.0)), ("cat", RandomCat([10]))]


@pytest.mark.parametrize(
    "func",
    [
        Debug(),
        RandomGaussian(),
        RandomBinary(),
        RandomSymmetricDirichlet(),
        BinaryMarkovChain(0.1, 0.1),
        Constant(1),
        LinearTrend(),
        RandomCat([10]),
        Lag("foo", 1),
        ForEachCat(RandomGaussian()),
        Eval("np.random.rand(length)"),
        SmoothSeasonality(Constant(12), Constant(0)),
        Add(["foo", "foo"]),
        Mul(["foo", "foo"]),
        NanWhere("foo", "foo"),
        Stack([Ref("foo"), Ref("foo")]),
        RandomGaussian() + RandomGaussian(),
        RandomGaussian() * RandomGaussian(),
        RandomGaussian() / RandomGaussian(),
    ],
)
def test_call_and_repr(func) -> None:
    global_state = {}
    x = evaluate(BASE_RECIPE, length=10, global_state=global_state)
    kwargs = dict(foo=42, bar=23)
    np.random.seed(0)
    ret = func(
        x,
        field_name="bar",
        length=10,
        global_state=global_state.copy(),
        **kwargs,
    )

    func_reconstructed = load_code(dump_code(func))

    np.random.seed(0)
    ret2 = func_reconstructed(
        x,
        field_name="foo",
        length=10,
        global_state=global_state.copy(),
        **kwargs,
    )
    print(ret)
    np.testing.assert_allclose(ret2, ret)


@pytest.mark.parametrize(
    "recipe",
    [
        [
            ("target", LinearTrend() + RandomGaussian()),
            ("binary_causal", BinaryMarkovChain(0.01, 0.1)),
            ("feat_dynamic_real", Stack([Ref("binary_causal")])),
            ("feat_static_cat", RandomCat([10])),
            (
                "feat_static_real",
                ForEachCat(RandomGaussian(1, (10,)), "feat_static_cat")
                + RandomGaussian(0.1, (10,)),
            ),
        ],
        lambda **kwargs: dict(
            target=np.random.rand(kwargs["length"]),
            feat_dynamic_real=np.random.rand(2, kwargs["length"]),
            feat_static_cat=[0],
            feat_static_real=[0.1, 0.2],
        ),
    ],
)
def test_recipe_dataset(recipe) -> None:
    data = RecipeDataset(
        recipe=recipe,
        metadata=MetaData(
            freq="D",
            feat_static_real=[BasicFeatureInfo(name="feat_static_real_000")],
            feat_static_cat=[
                CategoricalFeatureInfo(name="foo", cardinality=10)
            ],
            feat_dynamic_real=[BasicFeatureInfo(name="binary_causal")],
        ),
        max_train_length=20,
        prediction_length=10,
        num_timeseries=10,
        trim_length_fun=lambda x, **kwargs: np.minimum(
            int(np.random.geometric(1 / (kwargs["train_length"] / 2))),
            kwargs["train_length"],
        ),
    )

    generated = data.generate()
    generated_train = list(generated.train)
    generated_test = list(generated.test)
    train_lengths = np.array([len(x["target"]) for x in generated_train])
    test_lengths = np.array([len(x["target"]) for x in generated_test])
    assert np.all(test_lengths >= 10)
    assert np.all(test_lengths - train_lengths >= 10)

    assert len(list(generated.train)) == 10


@pytest.mark.parametrize("recipe", [BASE_RECIPE, lambda **kwargs: dict()])
def test_generate(recipe) -> None:
    start = pd.Timestamp("2014-01-01", freq="D")
    result = take_as_list(
        iterator=generate(length=10, recipe=BASE_RECIPE, start=start), num=10
    )
    assert len(result) == 10


def test_two() -> None:
    class Two(Lifted):
        num_outputs = 2

        @validated()
        def __init__(self):
            pass

        def __call__(self, x: Env, length: int, *args, **kwargs):
            return np.random.randn(length), np.random.randn(length)

    a, b = Two()
    evaluate(a, 100)


def test_functional() -> None:
    daily_smooth_seasonality = SmoothSeasonality(period=288, phase=-72)
    noise = RandomGaussian(stddev=0.1)
    signal = daily_smooth_seasonality + noise

    recipe = dict(
        daily_smooth_seasonality=daily_smooth_seasonality,
        noise=noise,
        signal=signal,
    )
    res = evaluate(recipe, length=100)
    for k in recipe.keys():
        assert k in res
        assert len(res[k]) == 100


def test_lifted_decorator() -> None:
    @rcp.lift
    def something(a, b, length):
        return np.concatenate([a[:10], b[:10]])

    noise1 = lnp.random.uniform(size=1000)
    noise2 = lnp.random.uniform(size=1000)
    res = something(noise1, noise2)
    length = rcp.Length(res)
    res = evaluate(res, length=length)
    assert len(res) == 20

    @rcp.lift(2)
    def something_else(a, b, length):
        return a[:10], b[:10]

    noise1 = lnp.random.uniform(size=1000)
    noise2 = lnp.random.uniform(size=1000)
    a, b = something_else(noise1, noise2)
    res = evaluate([a, b], length=length)
    assert len(res[0]) == 10
    assert len(res[1]) == 10


def test_length() -> None:
    u = rcp.Constant(np.array([1, 2, 3, 4, 5, 6, 7]))
    x = u * RandomGaussian()
    assert len(evaluate(x, length=rcp.Length(u))) == 7

    l = rcp.Length()
    assert evaluate(l, length=9) == 9


def test_arp() -> None:
    time_scale = 10 ** lnp.random.uniform(low=1, high=10)
    u = rcp.normalized_ar1(time_scale, norm="minmax")
    x = evaluate(u, length=1000)
    assert len(x) == 1000
    assert x.max() == 1
    assert x.min() == 0
