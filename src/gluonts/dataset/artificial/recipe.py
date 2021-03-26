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

import doctest
import functools
import itertools
import operator
import textwrap
from types import SimpleNamespace
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
    cast,
)

import numpy as np
import pandas as pd

from gluonts.core.component import validated
from gluonts.dataset.common import DataEntry

ValueOrCallable = Union[Any, Callable]
Recipe = Union[
    Callable, List[Callable], List[Tuple[str, Callable]], Dict[str, Callable]
]
Env = Dict[str, Any]


def resolve(val_or_callable: ValueOrCallable, context: Env, *args, **kwargs):
    if callable(val_or_callable):
        key = f"_ref_{id(val_or_callable)}"
        if key not in context:
            r = val_or_callable(context, *args, **kwargs)
            context[key] = r
            if "field_name" in kwargs:
                context[kwargs["field_name"]] = r
        return context[key]
    elif isinstance(val_or_callable, str):
        return context[val_or_callable]
    elif isinstance(val_or_callable, list):
        return [resolve(v, context, *args, **kwargs) for v in val_or_callable]
    elif isinstance(val_or_callable, tuple):
        return tuple(
            [resolve(v, context, *args, **kwargs) for v in val_or_callable]
        )
    else:
        return val_or_callable


def generate(
    length: int,
    recipe: Recipe,
    start: pd.Timestamp,
    global_state: Optional[dict] = None,
    seed: int = 0,
    item_id_prefix: str = "",
) -> Iterator[DataEntry]:
    np.random.seed(seed)

    if global_state is None:
        global_state = {}

    if isinstance(recipe, (dict, list)):
        for x in itertools.count():
            data = evaluate(recipe, length=length, global_state=global_state)
            yield dict(**data, item_id=item_id_prefix + str(x), start=start)
    else:
        assert callable(
            recipe
        ), "generate can only be used with dictionary recipes"
        for x in itertools.count():
            data = recipe(length=length, global_state=global_state)
            assert isinstance(
                data, dict
            ), "generate can only be used with dictionary recipes"
            yield dict(**data, item_id=item_id_prefix + str(x), start=start)


def evaluate(
    recipe: Recipe,
    length: ValueOrCallable,
    *args,
    global_state: dict = None,
    **kwargs,
) -> Any:
    if global_state is None:
        global_state = {}

    context = {}
    if "length" in kwargs:
        del kwargs["length"]
    if "field_name" in kwargs:
        del kwargs["field_name"]
    if "global_state" in kwargs:
        del kwargs["global_state"]
    if "context" in kwargs:
        context = kwargs["context"]
        del kwargs["context"]

    length_value = resolve(
        length,
        context,
        length=None,
        global_state=global_state,
        *args,
        **kwargs,
    )

    # convert previous format into dict
    if isinstance(recipe, list) and len(recipe) > 0:
        elem = recipe[0]
        if (
            isinstance(elem, (tuple, list))
            and len(elem) == 2
            and isinstance(elem[0], str)
        ):
            recipe = cast(List[Tuple[str, Callable]], recipe)
            recipe = dict(recipe)

    if isinstance(recipe, dict):
        data: DataEntry = {}
        for k, f in recipe.items():
            try:
                data[k] = resolve(
                    f,
                    context,
                    length=length_value,
                    field_name=k,
                    global_state=global_state,
                    *args,
                    **kwargs,
                )
            except ValueError as e:
                raise ValueError(f'Error while evaluating key "{k}"') from e
        return data
    elif callable(recipe):
        return resolve(
            recipe,
            context,
            length=length_value,
            global_state=global_state,
            *args,
            **kwargs,
        )
    elif isinstance(recipe, list):
        result = []
        for i, ri in enumerate(recipe):
            try:
                result.append(
                    resolve(
                        ri,
                        context,
                        length=length_value,
                        global_state=global_state,
                        *args,
                        **kwargs,
                    )
                )
            except ValueError as e:
                raise ValueError(
                    f"Error when evaluating entry {i} or recipe"
                ) from e
        return result
    else:
        raise ValueError(f"Invalid type for recipe {recipe}")


def make_func(
    length: int, recipe: Recipe, global_state=None
) -> Callable[[int, Env], DataEntry]:
    if global_state is None:
        global_state = {}

    def f(length=length, global_state=global_state, *args, **kwargs):
        return evaluate(
            recipe, length=length, global_state=global_state, *args, **kwargs
        )

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
    num_outputs: int = 1

    def __add__(self, other):
        return _LiftedBinaryOp(self, other, "+")

    def __radd__(self, other):
        return _LiftedBinaryOp(other, self, "+")

    def __sub__(self, other):
        return _LiftedBinaryOp(self, other, "-")

    def __rsub__(self, other):
        return _LiftedBinaryOp(other, self, "-")

    def __mul__(self, other):
        return _LiftedBinaryOp(self, other, "*")

    def __rmul__(self, other):
        return _LiftedBinaryOp(other, self, "*")

    def __truediv__(self, other):
        return _LiftedBinaryOp(self, other, "/")

    def __rtruediv__(self, other):
        return _LiftedBinaryOp(other, self, "/")

    def __pow__(self, other):
        return _LiftedBinaryOp(self, other, "**")

    def __rpow__(self, other):
        return _LiftedBinaryOp(other, self, "**")

    def __and__(self, other):
        return _LiftedBinaryOp(self, other, "&")

    def __rand__(self, other):
        return _LiftedBinaryOp(other, self, "&")

    def __or__(self, other):
        return _LiftedBinaryOp(self, other, "|")

    def __ror__(self, other):
        return _LiftedBinaryOp(other, self, "|")

    def __xor__(self, other):
        return _LiftedBinaryOp(self, other, "^")

    def __rxor__(self, other):
        return _LiftedBinaryOp(other, self, "^")

    def __gt__(self, other):
        return _LiftedBinaryOp(self, other, ">")

    def __ge__(self, other):
        return _LiftedBinaryOp(self, other, ">=")

    def __lt__(self, other):
        return _LiftedBinaryOp(self, other, "<")

    def __le__(self, other):
        return _LiftedBinaryOp(self, other, "<=")

    def __eq__(self, other):
        return _LiftedBinaryOp(self, other, "==")

    def __ne__(self, other):
        return _LiftedBinaryOp(self, other, "!=")

    def __iter__(self):
        for i in range(self.num_outputs):
            yield _LiftedUnpacked(self, i)

    def __call__(
        self,
        x: Env,
        length: int,
        field_name: str,
        global_state: Dict,
        *args,
        **kwargs,
    ):
        pass


def expand_shape(s, length):
    if isinstance(s, int):
        s = (s,)
    if s is None or len(s) == 0:
        return s
    s = np.array(s)
    if np.any(s == 0):
        s[s == 0] = length
    return tuple(s)


class NumpyFunc(Lifted):
    @validated()
    def __init__(
        self,
        func: str,
        func_args: Tuple[Any, ...],
        func_kwargs: Dict[str, Any],
    ):
        import numpy

        splits = func.split(".")
        b = numpy
        for s in splits:
            b = getattr(b, s)
        self.func = b
        self.func_args = func_args
        self.func_kwargs = func_kwargs

    def __call__(self, x: Env, length: int, *args, **kwargs):
        func_args = [
            resolve(u, x, length=length, *args, **kwargs)
            for u in self.func_args
        ]
        func_kwargs = {}
        for k, v in self.func_kwargs.items():
            v = resolve(v, x, length=length, *args, **kwargs)
            if k in ["shape", "size"]:
                v = expand_shape(v, length=length)
            func_kwargs[k] = v
        return self.func(*func_args, **func_kwargs)


lifted_numpy = SimpleNamespace()
lifted_numpy.random = SimpleNamespace()


_NUMPY_FUNC_NAMES = [
    "abs",
    "all",
    "any",
    "arange",
    "arccos",
    "arccosh",
    "arcsin",
    "arcsinh",
    "arctan",
    "arctan2",
    "arctanh",
    "argmax",
    "argmin",
    "ceil",
    "clip",
    "concatenate",
    "convolve",
    "cos",
    "cosh",
    "cumsum",
    "cumprod",
    "divmod",
    "exp",
    "floor",
    "floor_divide",
    "isfinite",
    "isinf",
    "isnan",
    "linspace",
    "log",
    "log10",
    "max",
    "mean",
    "mod",
    "min",
    "nan_to_num",
    "nanargmax",
    "nanargmin",
    "nancumsum",
    "nancumprod",
    "nanmax",
    "nanmean",
    "nanmin",
    "nanpercentile",
    "nanquantile",
    "nansum",
    "nanstd",
    "nanprod",
    "nanvar",
    "ones",
    "ones_like",
    "percentile",
    "power",
    "prod",
    "quantile",
    "random.beta",
    "random.binomial",
    "random.choice",
    "random.chisquare",
    "random.dirichlet",
    "random.exponential",
    "random.gamma",
    "random.geometric",
    "random.gumbel",
    "random.hypergeometric",
    "random.laplace",
    "random.logistic",
    "random.lognormal",
    "random.logseries",
    "random.multinomial",
    "random.negative_binomial",
    "random.normal",
    "random.pareto",
    "random.power",
    "random.poisson",
    "random.randn",
    "random.randint",
    "random.shuffle",
    "random.uniform",
    "random.vonmises",
    "random.weibull",
    "remainder",
    "repeat",
    "reshape",
    "round",
    "sin",
    "sinh",
    "shape",
    "stack",
    "std",
    "sum",
    "take",
    "tan",
    "tanh",
    "unique",
    "var",
    "where",
    "zeros",
    "zeros_like",
]


for func_name in _NUMPY_FUNC_NAMES:
    normalized_func_name = f"_np_shim_{func_name.replace('.', '_')}"
    if normalized_func_name in globals():
        continue

    s = f"""
    @functools.wraps(np.{func_name})
    def {normalized_func_name}(*args, **kwargs):
        return NumpyFunc('{func_name}', args, kwargs)

    lifted_numpy.{func_name} = {normalized_func_name}
    # disable numpy docstring tests
    lifted_numpy.{func_name}.__doc__ = lifted_numpy.{func_name}.__doc__.replace('>>>', '>>')
    """
    exec(textwrap.dedent(s))


class Length(Lifted):
    @validated()
    def __init__(self, l: ValueOrCallable = None):
        self.l = l

    def __call__(self, x: Env, length: int, *args, **kwargs):
        l = resolve(self.l, x, length, *args, **kwargs)
        if l is None:
            assert (
                length is not None
            ), "Cannot get value for Length() when length is not provided in evaluate"
            return length
        return len(l)


def lift(input: Union[int, Callable]):
    """
    Use this decorator to lift a function.

    @lift
    def f(x, y, length=None)

    or if your function returns more results

    @lift(2)
    def f(x, y, length=None)

    You can then use your function as part of a recipe. The function is called
    with all all arguments being already resolved.

    Note that you cannot serialize recipes that use the lift decorated functions.
    """
    if isinstance(input, int):
        num_outs = input
    else:
        num_outs = 1

    def w(f):
        @functools.wraps(f)
        def g(*f_args, **f_kwargs):
            class Tmp(Lifted):
                num_outputs = num_outs

                @validated()
                def __init__(self, f, f_args, f_kwargs):
                    self.f = f
                    self.f_args = f_args
                    self.f_kwargs = f_kwargs

                def __call__(self, x: Env, length: int, *args, **kwargs):
                    resolved_f_args = [
                        resolve(a, x, length, *args, **kwargs)
                        for a in self.f_args
                    ]
                    resolved_f_kwargs = {
                        k: resolve(v, x, length, *args, **kwargs)
                        for k, v in self.f_kwargs.items()
                    }
                    return self.f(
                        *resolved_f_args, **resolved_f_kwargs, length=length
                    )

            return Tmp(f, f_args, f_kwargs)

        return g

    if isinstance(input, int):
        return w
    else:
        return w(input)


class _LiftedUnpacked(Lifted):
    @validated()
    def __init__(self, base: Lifted, i: int):
        self.base = base
        self.i = i

    def __call__(self, *args, **kwargs):
        v = resolve(self.base, *args, **kwargs)
        return v[self.i]


class _LiftedBinaryOp(Lifted):
    @validated()
    def __init__(self, left, right, op) -> None:
        self.left = left
        self.right = right
        self.op = {
            "+": operator.add,
            "*": operator.mul,
            "-": operator.sub,
            "/": operator.truediv,
            "**": operator.pow,
            ">": operator.gt,
            ">=": operator.ge,
            "<": operator.lt,
            "<=": operator.le,
            "==": operator.eq,
            "!=": operator.ne,
            "|": operator.or_,
            "&": operator.and_,
            "^": operator.xor,
        }[op]

    def __call__(self, *args, **kwargs):
        left = resolve(self.left, *args, **kwargs)
        right = resolve(self.right, *args, **kwargs)
        return self.op(left, right)


class RandomGaussian(Lifted):
    @validated()
    def __init__(
        self, stddev: ValueOrCallable = 1.0, shape: Sequence[int] = (0,)
    ) -> None:
        self.stddev = stddev
        self.shape = shape

    def __call__(self, x: Env, length: int, *args, **kwargs):
        stddev = resolve(self.stddev, x, length, *args, **kwargs)
        s = expand_shape(self.shape, length)
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
        s = expand_shape(self.shape, length)
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
        inp = resolve(self.input, x, *args, **kwargs)
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
        **kwargs,
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


_LEGACY_WARNING_WAS_SHOWN = False


class Ref(Lifted):
    @validated()
    def __init__(self, field_name: str) -> None:
        global _LEGACY_WARNING_WAS_SHOWN
        if not _LEGACY_WARNING_WAS_SHOWN:
            import warnings

            warnings.warn(
                "Ref is deprecated. Please use the functional api.",
            )
            _LEGACY_WARNING_WAS_SHOWN = True
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
        s = expand_shape(self.shape, length)
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
        s = expand_shape(self.shape, length)
        return np.random.randint(low, high, s)


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


class ARp(Lifted):
    def __init__(
        self,
        phi: ValueOrCallable,
        sigma: ValueOrCallable,
        xhist: ValueOrCallable = None,
        c: ValueOrCallable = 0.0,
        noise: ValueOrCallable = None,
    ):
        """
        Draw samples from an ARp process.
        Parametrized as in

        https://en.wikipedia.org/wiki/Autoregressive_model#Graphs_of_AR(p)_processes
        """
        self.phi = phi
        self.sigma = sigma
        self.c = c
        self.xhist = xhist
        self.noise = noise

    def __call__(self, x: Env, length: int, *args, **kwargs):
        from .ar_p import ar_p

        phi = resolve(self.phi, x, length, *args, **kwargs)
        phi = np.asarray(phi, dtype=np.float64)
        assert phi.ndim == 1, "phi should be a 1d array"
        assert len(phi) > 0, "phi should have length > 0"
        sigma = resolve(self.sigma, x, length, *args, **kwargs)
        sigma = float(sigma)
        xhist = resolve(self.xhist, x, length, *args, **kwargs)
        if xhist is None:
            xhist = np.zeros_like(phi)
        xhist = np.asarray(xhist, dtype=np.float64)
        assert (
            xhist.shape == phi.shape
        ), "xhist should have the same length as phi"
        c = resolve(self.c, x, length, *args, **kwargs)
        noise = resolve(self.noise, x, length, *args, **kwargs)
        if noise is not None:
            noise = np.asarray(noise, dtype=np.float64)
            assert noise.ndim == 1, "the noise should be a 1d array"
            assert (
                len(noise) == length
            ), f"len(noise) should be be length={length}"

        v = ar_p(
            phi=phi,
            sigma=sigma,
            length=length,
            xhist=xhist,
            c=c,
            noise=noise,
        )
        return v


def normalized_ar1(tau, x0=None, norm="minmax", sigma=1.0):
    r"""
    Returns an ar1 process with an auto correlation time of tau.

    norm can be
    None -> no normalization
    'minmax' -> min_max_scaled
    'standard' -> 0 mean, unit variance
    """
    assert norm in [None, "minmax", "standard"]
    phi = lifted_numpy.exp(-1.0 / tau)
    a = ARp(phi=[phi], xhist=[x0] if x0 is not None else None, sigma=sigma)

    if norm is None:
        return a
    elif norm == "minmax":
        amin = lifted_numpy.min(a)
        amax = lifted_numpy.max(a)
        return (a - amin) / (amax - amin)
    elif norm == "standard":
        return (a - lifted_numpy.mean(a)) / lifted_numpy.std(a)
    else:
        raise NotImplementedError()


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
    def __init__(
        self, recipe: Recipe, op: Optional[ValueOrCallable] = None
    ) -> None:
        self.recipe = recipe
        self.op = op

    def __call__(self, x: Env, *args, **kwargs):
        xx = evaluate(self.recipe, *args, **kwargs)
        if self.op is not None:
            return resolve(self.op, xx, *args, **kwargs)
        else:
            return xx
