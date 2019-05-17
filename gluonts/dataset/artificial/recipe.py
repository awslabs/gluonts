# Standard library imports
import functools
import itertools
import operator
from typing import Any, Callable, Iterator, List, Optional, Tuple, Union

# Third-party imports
import numpy as np
import pandas as pd

# First-party imports
from gluonts.core.component import validated
from gluonts.dataset.common import DataEntry


def generate(
    length: int,
    recipe: Union[Callable, List[Tuple[str, Callable]]],
    start: pd.Timestamp,
    global_state: Optional[dict] = None,
    seed: int = 0,
) -> Iterator[DataEntry]:
    np.random.seed(seed)

    if global_state is None:
        global_state = {}

    if isinstance(recipe, list):
        for x in itertools.count():
            data: DataEntry = {}
            for k, f in recipe:
                data[k] = f(
                    data,
                    length=length,
                    field_name=k,
                    global_state=global_state,
                )
            yield dict(**data, item=x, start=start)
    else:
        assert callable(recipe)
        for x in itertools.count():
            data = recipe(length=length, global_state=global_state)
            yield dict(**data, item=x, start=start)


def evaluate_recipe(
    funcs: List[Tuple[str, Callable]], length: int, global_state: dict = None
) -> dict:
    if global_state is None:
        global_state = {}
    data: DataEntry = {}
    for k, f in funcs:
        data[k] = f(
            data, length=length, field_name=k, global_state=global_state
        )
    return data


def make_func(
    length: int, funcs: List[Tuple[str, Callable]], global_state=None
) -> Callable[[int, dict], DataEntry]:
    if global_state is None:
        global_state = {}

    def f(length=length, global_state=global_state):
        data = {}
        for k, f in funcs:
            data[k] = f(
                data, length=length, field_name=k, global_state=global_state
            )
        return data

    return f


def take_as_list(iterator, num):
    return list(itertools.islice(iterator, num))


class Debug:
    @validated()
    def __init__(self, print_global=False):
        self.print_global = print_global

    def __call__(self, x, global_state, **kwargs):
        print(x)
        if self.print_global:
            print(global_state)
        return 0


class Lifted:
    def __add__(self, other):
        return LiftedAdd(self, other)

    def __radd__(self, other):
        return LiftedAdd(other, self)

    def __mul__(self, other):
        return LiftedMul(self, other, operator.mul)

    def __rmul__(self, other):
        return LiftedMul(other, self, operator.mul)

    def __truediv__(self, other):
        return LiftedTruediv(self, other, operator.truediv)

    def __rtruediv__(self, other):
        return LiftedTruediv(other, self, operator.truediv)


class LiftedBinaryOp(Lifted):
    def __init__(self, left, right, op) -> None:
        self.left = left
        self.right = right
        self.op = op

    def __call__(self, *args, **kwargs):
        if callable(self.left):
            left = self.left(*args, **kwargs)
        else:
            left = self.left
        if callable(self.right):
            right = self.right(*args, **kwargs)
        else:
            right = self.right
        return self.op(left, right)


class LiftedAdd(LiftedBinaryOp):
    @validated()
    def __init__(self, left, right) -> None:
        super().__init__(left, right, operator.add)


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
        self, stddev: float = 1.0, length: Optional[int] = None
    ) -> None:
        self.stddev = stddev
        self.length = length

    def __call__(self, x, length, **kwargs):
        if self.length is not None:
            length = self.length
        return self.stddev * np.random.randn(length)


# Binary recipe that returns 1 if date is in holidays list and 0 otherwise
class Binary(Lifted):
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
    def __init__(self, prob: float = 0.1) -> None:
        self.prob = prob

    def __call__(self, x, length, **kwargs):
        return 1.0 * (np.random.rand(length) < self.prob)


class RandomSymmetricDirichlet(Lifted):
    @validated()
    def __init__(
        self, alpha: float = 1.0, length: Optional[int] = None
    ) -> None:
        self.alpha = alpha
        self.length = length

    def __call__(self, x, length, **kwargs):
        if self.length is not None:
            length = self.length

        return np.random.dirichlet(self.alpha * np.ones(length))


class BinaryMarkovChain(Lifted):
    @validated()
    def __init__(self, one_to_zero: float, zero_to_one: float) -> None:
        self.probs = np.zeros(2)
        self.probs[0] = zero_to_one
        self.probs[1] = one_to_zero

    def __call__(self, x, length, **kwargs):
        out = np.ones(length, dtype=np.int)  # initial state is 1
        uu = np.random.rand(length)
        for i in range(1, length):
            if uu[i] < self.probs[out[i - 1]]:
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
    def __init__(self, constant) -> None:
        self.constant = constant

    def __call__(self, x, length, *args, **kwargs):
        return self.constant * np.ones(length)


class LinearTrend(Lifted):
    @validated()
    def __init__(self, slope_fun: Callable = Constant(1.0)) -> None:
        self.slope_fun = slope_fun

    def __call__(self, x, length, **kwargs):
        return self.slope_fun() * np.arange(length) / length


class RandomCat:
    @validated()
    def __init__(
        self,
        cardinalities: List[int],
        prob_fun: Lifted = RandomSymmetricDirichlet(),
    ) -> None:
        self.cardinalities = cardinalities
        self.prob_fun = prob_fun

    def __call__(self, x, field_name, global_state, **kwargs):
        if field_name not in global_state:
            probs = [self.prob_fun(x, length=c) for c in self.cardinalities]
            global_state[field_name] = probs
        probs = global_state[field_name]
        cats = [
            np.random.choice(np.arange(len(probs[i])), p=probs[i])
            for i in range(len(probs))
        ]
        return cats


class Lag(Lifted):
    @validated()
    def __init__(self, field_name: str, lag: int = 0) -> None:
        self.field_name = field_name
        self.lag = lag

    def __call__(self, x, **kwargs):
        feat = x[self.field_name]
        if self.lag != 0:
            lagged_feat = np.concatenate(
                (np.zeros(self.lag), feat[: -self.lag])
            )
        else:
            lagged_feat = feat
        return lagged_feat


class ForEachCat(Lifted):
    @validated()
    def __init__(self, fun, cat_field='cat', cat_idx=0) -> None:
        self.fun = fun
        self.cat_field = cat_field
        self.cat_idx = cat_idx

    def __call__(self, x, field_name, global_state, **kwargs):
        c = x[self.cat_field][self.cat_idx]
        if field_name not in global_state:
            global_state[field_name] = np.empty(
                len(global_state[self.cat_field][self.cat_idx]),
                dtype=np.object,
            )
        if global_state[field_name][c] is None:
            global_state[field_name][c] = self.fun(x, **kwargs)
        return global_state[field_name][c]


class Expr(Lifted):
    @validated()
    def __init__(self, expr: str) -> None:
        self.expr = expr

    def __call__(self, x, **kwargs):
        return eval(self.expr, globals(), dict(x=x, **kwargs))


class SmoothSeasonality(Lifted):
    @validated()
    def __init__(self, period_fun: Callable, phase_fun: Callable) -> None:
        self.period_fun = period_fun
        self.phase_fun = phase_fun

    def __call__(self, x, length, **kwargs):
        return (
            np.sin(
                2.0
                / self.period_fun()
                * np.pi
                * (np.arange(length) + self.phase_fun())
            )
            + 1
        ) / 2.0


class Add(Lifted):
    @validated()
    def __init__(self, inputs) -> None:
        self.inputs = inputs

    def __call__(self, x, **kwargs):
        return sum([x[k] for k in self.inputs])


class Mul(Lifted):
    @validated()
    def __init__(self, inputs) -> None:
        self.inputs = inputs

    def __call__(self, x, **kwargs):
        return functools.reduce(operator.mul, [x[k] for k in self.inputs])


class NanWhere(Lifted):
    @validated()
    def __init__(self, source_name, nan_indicator_name) -> None:
        self.source_name = source_name
        self.nan_indicator_name = nan_indicator_name

    def __call__(self, x, **kwargs):
        out = x[self.source_name]
        out[x[self.nan_indicator_name] == 1] = np.nan
        return out


class NanWhereNot(Lifted):
    @validated()
    def __init__(self, source_name, nan_indicator_name) -> None:
        self.source_name = source_name
        self.nan_indicator_name = nan_indicator_name

    def __call__(self, x, **kwargs):
        out = x[self.source_name]
        out[x[self.nan_indicator_name] == 0] = np.nan
        return out


class Stack(Lifted):
    @validated()
    def __init__(self, inputs: List[str]) -> None:
        self.inputs = inputs

    def __call__(self, x, **kwargs):
        return np.stack([x[k] for k in self.inputs], axis=0)
