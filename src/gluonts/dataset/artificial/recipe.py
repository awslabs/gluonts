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
import functools
import itertools
import operator
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

# Third-party imports
import numpy as np
import pandas as pd

# First-party imports
from gluonts.core.component import validated
from gluonts.dataset.common import DataEntry

ValueOrCallable = Union[Any, Callable]
Recipe = List[Tuple[str, Callable]]
Env = Dict[str, Any]


def resolve(val_or_callable: ValueOrCallable, context: Env, *args, **kwargs):
    if callable(val_or_callable):
        return val_or_callable(context, *args, **kwargs)
    elif isinstance(val_or_callable, str):
        return context[val_or_callable]
    else:
        return val_or_callable


def generate(
    length: int,
    recipe: Union[Callable, Recipe],
    start: pd.Timestamp,
    global_state: Optional[dict] = None,
    seed: int = 0,
    item_id_prefix: str = "",
) -> Iterator[DataEntry]:
    np.random.seed(seed)

    if global_state is None:
        global_state = {}

    if isinstance(recipe, list):
        for x in itertools.count():
            data: DataEntry = {}
            for k, f in recipe:
                data[k] = resolve(
                    f,
                    data,
                    length=length,
                    field_name=k,
                    global_state=global_state,
                )
            yield dict(**data, item_id=item_id_prefix + str(x), start=start)
    else:
        assert callable(recipe)
        for x in itertools.count():
            data = recipe(length=length, global_state=global_state)
            yield dict(**data, item_id=item_id_prefix + str(x), start=start)


def evaluate(
    funcs: Recipe, length: int, *args, global_state: dict = None, **kwargs
) -> Env:
    if global_state is None:
        global_state = {}

    if "length" in kwargs:
        del kwargs["length"]
    if "field_name" in kwargs:
        del kwargs["field_name"]
    if "global_state" in kwargs:
        del kwargs["global_state"]

    data: DataEntry = {}
    for k, f in funcs:
        try:
            data[k] = resolve(
                f,
                data,
                length=length,
                field_name=k,
                global_state=global_state,
                *args,
                **kwargs
            )
        except ValueError as e:
            raise ValueError('Error while evaluating key "{}"'.format(k), e)

    return data


def make_func(
    length: int, funcs: Recipe, global_state=None
) -> Callable[[int, Env], DataEntry]:
    if global_state is None:
        global_state = {}

    def f(length=length, global_state=global_state, *args, **kwargs):
        data = {}
        for k, f in funcs:
            data[k] = resolve(
                f,
                data,
                length=length,
                field_name=k,
                global_state=global_state,
                *args,
                **kwargs
            )
        return data

    return f


def take_as_list(iterator, num):
    return list(itertools.islice(iterator, num))


class Debug:
    @validated()
    def __init__(self, print_global=False) -> None:
        self.print_global = print_global

    def __call__(self, x: Env, global_state, **kwargs):
        print(x)
        if self.print_global:
            print(global_state)
        return 0


class Lifted:
    def __add__(self, other):
        return LiftedAdd(self, other)

    def __radd__(self, other):
        return LiftedAdd(other, self)

    def __sub__(self, other):
        return LiftedSub(self, other)

    def __rsub__(self, other):
        return LiftedSub(other, self)

    def __mul__(self, other):
        return LiftedMul(self, other)

    def __rmul__(self, other):
        return LiftedMul(other, self)

    def __truediv__(self, other):
        return LiftedTruediv(self, other)

    def __rtruediv__(self, other):
        return LiftedTruediv(other, self)

    def __pow__(self, other):
        return LiftedBinaryOp(self, other, operator.pow)

    def __call__(
        self,
        x: Env,
        length: int,
        field_name: str,
        global_state: Dict,
        *args,
        **kwargs
    ):
        pass


class LiftedBinaryOp(Lifted):
    def __init__(self, left, right, op) -> None:
        self.left = left
        self.right = right
        self.op = op

    def __call__(self, *args, **kwargs):
        left = resolve(self.left, *args, **kwargs)
        right = resolve(self.right, *args, **kwargs)
        return self.op(left, right)


class LiftedAdd(LiftedBinaryOp):
    @validated()
    def __init__(self, left, right) -> None:
        super().__init__(left, right, operator.add)


class LiftedSub(LiftedBinaryOp):
    @validated()
    def __init__(self, left, right) -> None:
        super().__init__(left, right, operator.sub)


class LiftedMul(LiftedBinaryOp):
    @validated()
    def __init__(self, left, right) -> None:
        super().__init__(left, right, operator.mul)


class LiftedTruediv(LiftedBinaryOp):
    @validated()
    def __init__(self, left, right) -> None:
        super().__init__(left, right, operator.truediv)


class RandomGaussian(Lifted):
    @validated()
    def __init__(
        self, stddev: ValueOrCallable = 1.0, shape: Sequence[int] = (0,)
    ) -> None:
        self.stddev = stddev
        self.shape = shape

    def __call__(self, x: Env, length: int, *args, **kwargs):
        stddev = resolve(self.stddev, x, length, *args, **kwargs)
        s = np.array(self.shape)
        s[s == 0] = length
        return stddev * np.random.randn(*s)


# Binary recipe that returns 1 if date is in holidays list and 0 otherwise
class BinaryHolidays(Lifted):
    @validated()
    # TODO: holidays is type List[datetime.date]
    def __init__(self, dates: List[pd.Timestamp], holidays: List[Any]) -> None:
        self.dates = dates
        self.holidays = holidays

    def __call__(self, *args, **kwargs):
        length = len(self.dates)
        out = np.ones(length)
        for i, date in enumerate(self.dates):
            # Convert to string to check if inside of holidays datatime.date
            if date.date() in self.holidays:
                out[i] = 1.0
            else:
                out[i] = 0.0
        return out


class RandomBinary(Lifted):
    @validated()
    def __init__(self, prob: ValueOrCallable = 0.1) -> None:
        self.prob = prob

    def __call__(self, x: Env, length: int, *args, **kwargs):
        prob = resolve(self.prob, x, length, *args, **kwargs)
        return 1.0 * (np.random.rand(length) < prob)


class RandomSymmetricDirichlet(Lifted):
    @validated()
    def __init__(
        self, alpha: ValueOrCallable = 1.0, shape: Sequence[int] = (0,)
    ) -> None:
        self.alpha = alpha
        self.shape = shape

    def __call__(self, x, length, *args, **kwargs):
        alpha = resolve(self.alpha, x, length, *args, **kwargs)
        s = np.array(self.shape)
        s[s == 0] = length
        return np.random.dirichlet(alpha * np.ones(s))


class BinaryMarkovChain(Lifted):
    @validated()
    def __init__(
        self, one_to_zero: ValueOrCallable, zero_to_one: ValueOrCallable
    ) -> None:
        self.one_to_zero = one_to_zero
        self.zero_to_one = zero_to_one

    def __call__(self, x: Env, length: int, *args, **kwargs):
        probs = np.zeros(2)
        probs[0] = resolve(self.zero_to_one, x, length, *args, **kwargs)
        probs[1] = resolve(self.one_to_zero, x, length, *args, **kwargs)
        out = np.ones(length, dtype=np.int)  # initial state is 1
        uu = np.random.rand(length)
        for i in range(1, length):
            if uu[i] < probs[out[i - 1]]:
                out[i] = 1 - out[i - 1]
            else:
                out[i] = out[i - 1]
        return out


class Constant(Lifted):
    @validated()
    def __init__(self, constant) -> None:
        self.constant = constant

    def __call__(self, *args, **kwargs):
        return self.constant


class ConstantVec(Lifted):
    @validated()
    def __init__(self, constant: ValueOrCallable) -> None:
        self.constant = constant

    def __call__(self, x: Env, length: int, *args, **kwargs):
        constant = resolve(self.constant, x, length, *args, **kwargs)
        return constant * np.ones(length)


class NormalizeMax(Lifted):
    @validated()
    def __init__(self, input) -> None:
        self.input = input

    def __call__(self, x: Env, *args, **kwargs):
        inp = resolve(self.input, x, *args, kwargs)
        return inp / np.max(inp)


class OnesLike(Lifted):
    @validated()
    def __init__(self, other) -> None:
        self.other = other

    def __call__(self, x, length, *args, **kwargs):
        other = resolve(self.other, x, length, **kwargs)
        return np.ones_like(other)


class LinearTrend(Lifted):
    @validated()
    def __init__(self, slope: ValueOrCallable = 1.0) -> None:
        self.slope = slope

    def __call__(self, x, length, *args, **kwargs):
        slope = resolve(self.slope, x, length, *args, **kwargs)
        return slope * np.arange(length) / length


class RandomCat:
    @validated()
    def __init__(
        self,
        cardinalities: List[int],
        prob_fun: Callable = RandomSymmetricDirichlet(alpha=1.0, shape=(0,)),
    ) -> None:
        self.cardinalities = cardinalities
        self.prob_fun = prob_fun

    def __call__(self, x, field_name, global_state, **kwargs):
        if field_name not in global_state:
            probs = [self.prob_fun(x, length=c) for c in self.cardinalities]
            global_state[field_name] = probs
        probs = global_state[field_name]
        cats = np.array(
            [
                np.random.choice(np.arange(len(probs[i])), p=probs[i])
                for i in range(len(probs))
            ]
        )
        return cats


class Lag(Lifted):
    @validated()
    def __init__(
        self,
        input: ValueOrCallable,
        lag: ValueOrCallable = 0,
        pad_const: int = 0,
    ) -> None:
        self.input = input
        self.lag = lag
        self.pad_const = pad_const

    def __call__(self, x, *args, **kwargs):
        feat = resolve(self.input, x, *args, **kwargs)
        lag = resolve(self.lag, x, *args, **kwargs)

        if lag > 0:
            lagged_feat = np.concatenate(
                (self.pad_const * np.ones(lag), feat[:-lag])
            )
        elif lag < 0:
            lagged_feat = np.concatenate(
                (feat[-lag:], self.pad_const * np.ones(-lag))
            )

        else:
            lagged_feat = feat
        return lagged_feat


class ForEachCat(Lifted):
    @validated()
    def __init__(self, fun, cat_field="cat", cat_idx=0) -> None:
        self.fun = fun
        self.cat_field = cat_field
        self.cat_idx = cat_idx

    def __call__(
        self,
        x: Env,
        length: int,
        field_name: str,
        global_state: Dict,
        *args,
        **kwargs
    ):
        c = x[self.cat_field][self.cat_idx]
        if field_name not in global_state:
            global_state[field_name] = np.empty(
                len(global_state[self.cat_field][self.cat_idx]),
                dtype=np.object,
            )
        if global_state[field_name][c] is None:
            global_state[field_name][c] = self.fun(
                x, length=length, field_name=field_name, *args, **kwargs
            )
        return global_state[field_name][c]


class Eval(Lifted):
    @validated()
    def __init__(self, expr: str) -> None:
        self.expr = expr

    def __call__(self, x: Env, length: int, *args, **kwargs):
        return eval(self.expr, globals(), dict(x=x, length=length, **kwargs))


class SmoothSeasonality(Lifted):
    @validated()
    def __init__(
        self, period: ValueOrCallable, phase: ValueOrCallable
    ) -> None:
        self.period = period
        self.phase = phase

    def __call__(self, x: Env, length: int, *args, **kwargs):
        period = resolve(self.period, x, length, *args, **kwargs)
        phase = resolve(self.phase, x, length, *args, **kwargs)
        return (
            np.sin(2.0 / period * np.pi * (np.arange(length) + phase)) + 1
        ) / 2.0


class Add(Lifted):
    @validated()
    def __init__(self, inputs: List[ValueOrCallable]) -> None:
        self.inputs = inputs

    def __call__(self, x: Env, length: int, *args, **kwargs):
        return sum(
            [resolve(k, x, length, *args, **kwargs) for k in self.inputs]
        )


class Mul(Lifted):
    @validated()
    def __init__(self, inputs) -> None:
        self.inputs = inputs

    def __call__(self, x: Env, length: int, *args, **kwargs):
        return functools.reduce(
            operator.mul,
            [resolve(k, x, length, *args, **kwargs) for k in self.inputs],
        )


class NanWhere(Lifted):
    @validated()
    def __init__(
        self, source: ValueOrCallable, nan_indicator: ValueOrCallable
    ) -> None:
        self.source = source
        self.nan_indicator = nan_indicator

    def __call__(self, x: Env, length: int, *args, **kwargs):
        source = resolve(self.source, x, length, *args, **kwargs)
        nan_indicator = resolve(self.nan_indicator, x, length, *args, **kwargs)
        out = source.copy()
        out[nan_indicator == 1] = np.nan
        return out


class OneMinus(Lifted):
    @validated()
    def __init__(self, source: ValueOrCallable) -> None:
        self.source = source

    def __call__(self, x: Env, length: int, *args, **kwargs):
        value = resolve(self.source, x, length, *args, **kwargs)
        return 1 - value


class Concatenate(Lifted):
    @validated()
    def __init__(self, inputs: List[ValueOrCallable], axis: int = 0) -> None:
        self.inputs = inputs
        self.axis = axis

    def __call__(self, x: Env, length: int, *args, **kwargs):
        inputs = [resolve(z, x, length, **kwargs) for z in self.inputs]
        return np.concatenate(inputs, self.axis)


class Stack(Lifted):
    @validated()
    def __init__(self, inputs: List[ValueOrCallable]) -> None:
        self.inputs = inputs

    def __call__(self, x: Env, length: int, *args, **kwargs):
        inputs = [resolve(z, x, length, **kwargs) for z in self.inputs]
        return np.stack(inputs, axis=0)


class StackPrefix(Lifted):
    @validated()
    def __init__(self, prefix: str) -> None:
        self.prefix = prefix

    def __call__(self, x: Env, length: int, *args, **kwargs):
        inputs = [v for k, v in x.items() if k.startswith(self.prefix)]
        return np.stack(inputs, axis=0)


class Ref(Lifted):
    @validated()
    def __init__(self, field_name: str) -> None:
        self.field_name = field_name

    def __call__(self, x: Env, length: int, *args, **kwargs):
        return x[self.field_name]


class RandomUniform(Lifted):
    @validated()
    def __init__(
        self,
        low: ValueOrCallable = 0.0,
        high: ValueOrCallable = 1.0,
        shape=(0,),
    ) -> None:
        self.low = low
        self.high = high
        self.shape = shape

    def __call__(self, x: Env, length: int, *args, **kwargs):
        low = resolve(self.low, x, length, *args, **kwargs)
        high = resolve(self.high, x, length, *args, **kwargs)
        s = np.array(self.shape)
        s[s == 0] = length
        return np.random.uniform(low, high, s)


class RandomInteger(Lifted):
    @validated()
    def __init__(
        self,
        low: ValueOrCallable,
        high: ValueOrCallable,
        shape: Optional[Sequence[int]] = (0,),
    ) -> None:
        self.low = low
        self.high = high
        self.shape = shape

    def __call__(self, x: Env, length: int, *args, **kwargs):
        low = resolve(self.low, x, length, *args, **kwargs)
        high = resolve(self.high, x, length, *args, **kwargs)
        if self.shape is not None:
            s = np.array(self.shape)
            s[s == 0] = length
            return np.random.randint(low, high, s)
        else:
            return np.random.randint(low, high)


class RandomChangepoints(Lifted):
    @validated()
    def __init__(self, max_num_changepoints: ValueOrCallable) -> None:
        self.max_num_changepoints = max_num_changepoints

    def __call__(self, x: Env, length: int, *args, **kwargs):
        max_num_changepoints = resolve(
            self.max_num_changepoints, x, length, *args, **kwargs
        )
        num_changepoints = np.random.randint(0, max_num_changepoints + 1)
        change_idx = np.sort(
            np.random.randint(low=1, high=length - 1, size=(num_changepoints,))
        )
        change_ranges = np.concatenate([change_idx, [length]])
        out = np.zeros(length, dtype=np.int)
        for i in range(0, num_changepoints):
            out[change_ranges[i] : change_ranges[i + 1]] = i + 1
        return out


class Repeated(Lifted):
    @validated()
    def __init__(self, pattern: ValueOrCallable) -> None:
        self.pattern = pattern

    def __call__(self, x: Env, length: int, *args, **kwargs):
        pattern = resolve(self.pattern, x, length, **kwargs)
        repeats = length // len(pattern) + 1
        out = np.tile(pattern, (repeats,))
        return out[:length]


class Convolve(Lifted):
    @validated()
    def __init__(
        self, input: ValueOrCallable, filter: ValueOrCallable
    ) -> None:
        self.filter = filter
        self.input = input

    def __call__(self, x: Env, length: int, *args, **kwargs):
        fil = resolve(self.filter, x, length, **kwargs)
        inp = resolve(self.input, x, length, **kwargs)
        out = np.convolve(inp, fil, mode="same")
        return out


class Dilated(Lifted):
    @validated()
    def __init__(self, source: Callable, dilation: int) -> None:
        self.source = source
        self.dilation = dilation

    def __call__(self, x: Env, length: int, *args, **kwargs):
        inner = self.source(x, length // self.dilation + 1, **kwargs)
        out = np.repeat(inner, self.dilation)
        return out[:length]


class Choose(Lifted):
    @validated()
    def __init__(
        self, options: ValueOrCallable, selector: ValueOrCallable
    ) -> None:
        self.options = options
        self.selector = selector

    def __call__(self, x, length, **kwargs):
        options = resolve(self.options, x, length, **kwargs)
        selector = resolve(self.selector, x, length, **kwargs)
        e = np.eye(options.shape[0])
        out = np.sum(e[selector] * options.T, axis=1)
        return out


class EvalRecipe(Lifted):
    @validated()
    def __init__(self, recipe: Recipe, op: ValueOrCallable) -> None:
        self.recipe = recipe
        self.op = op

    def __call__(self, x: Env, *args, **kwargs):
        xx = evaluate(self.recipe, *args, **kwargs)
        return resolve(self.op, xx, *args, **kwargs)
