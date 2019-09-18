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

# Third-party imports
import numpy as np
import pandas as pd
import pytest

# First-party imports
from gluonts.core.serde import dump_code, load_code
from gluonts.dataset.common import (
    BasicFeatureInfo,
    CategoricalFeatureInfo,
    MetaData,
)
from gluonts.dataset.artificial import RecipeDataset
from gluonts.dataset.artificial.recipe import (
    Add,
    BinaryMarkovChain,
    Constant,
    ConstantVec,
    Debug,
    Eval,
    ForEachCat,
    Lag,
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
    take_as_list,
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
