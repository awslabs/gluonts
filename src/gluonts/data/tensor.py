from dataclasses import dataclass
from numbers import Number

import numpy as np


class Ax:
    def __init__(self, tensor):
        for dimension in tensor.dims:
            setattr(self, dimension, Slicer(tensor, dimension))


@dataclass
class Slicer:
    xs: object
    dim: str

    def __getitem__(self, idx):
        return self.xs.slice(self.dim, idx)


class Tensor(np.lib.mixins.NDArrayOperatorsMixin):
    __slots__ = "ax", "values", "shape"

    _sub_types = {}

    def __class_getitem__(cls, dims):
        if not isinstance(dims, tuple):
            dims = (dims,)

        if dims not in cls._sub_types:
            cls._sub_types[dims] = type(
                f"Tensor[{dims!r}]",
                (Tensor,),
                {
                    "dims": dict(zip(dims, range(len(dims)))),
                },
            )

        return cls._sub_types[dims]

    @classmethod
    def without_dims(cls, dimensions):
        return Tensor[tuple(cls.dims.keys() - set(dimensions))]

    @classmethod
    def new(cls, values):
        return cls(values)

    def __new__(cls, *args, **kwargs):
        assert cls is not Tensor
        return object.__new__(cls)

    def __init__(self, values):
        self.values = values
        self.ax = Ax(self)

        assert values.ndim == self.ndim
        self.shape = {
            name: self.values.shape[axis] for name, axis in self.dims.items()
        }

    @property
    def ndim(self):
        return len(self.__class__.dims)

    def unwrap(self, dims=None):
        if dims is None or dims == self.dims:
            return self.values

        return np.transpose(self.values, tuple(self.dims[dim] for dim in dims))

    def __repr__(self):
        shape = ", ".join(f"{dim}={n}" for dim, n in self.shape.items())
        return f"Tensor<{shape}>"

    def _call(self, ufunc, inputs, kwargs):
        if len(inputs) == 1:
            return self.new(ufunc(self.unwrap(), *kwargs))

        xs = []
        self_idx = None

        for idx, other in enumerate(inputs):
            if other is self:
                self_idx = idx

            if isinstance(other, (Number, Tensor)):
                xs.append(other)
            else:
                return NotImplementedError()

        if len(xs) == 2:
            if all(isinstance(x, Tensor) for x in xs):
                left, right, dims = unwrap_two(xs[0], xs[1])
                return Tensor["".join(dims)](ufunc(left, right, **kwargs))

            elif self_idx == 0:
                return self.__class__(ufunc(self.unwrap(), xs[1], *kwargs))
            else:
                return self.__class__(ufunc(xs[0], self.unwrap(), *kwargs))

        return self

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method == "__call__":
            return self._call(ufunc, inputs, kwargs)

        elif method == "reduce":
            axis = kwargs["axis"]
            if axis is None:
                return ufunc.reduce(self.unwrap(), **kwargs)

            kwargs["axis"] = self.dims[axis]
            return self.without_dims(axis)(ufunc.reduce(self.values, **kwargs))

        raise NotImplementedError()

    # def __getitem__(self, idx):
    #     return Tensor(index=self.index[idx], values=self.values[..., idx])

    def slice(self, axis, arg):
        idx = [slice(None)] * len(self.dims)
        idx[self.dims[axis]] = arg
        idx = tuple(idx)
        values = self.values[idx]

        if not isinstance(arg, slice):
            cls = self.without_dims(axis)
        else:
            cls = self.__class__

        return cls(values)


def unwrap_two(a, b):
    if len(a.shape) > len(b.shape):
        big = a
        small = b
    else:
        big = b
        small = a

    additional = small.shape.keys() - big.shape.keys()
    assert not additional

    order = list(big.shape.keys() - small.shape.keys()) + list(
        small.shape.keys()
    )

    if a is big:
        return a.unwrap(order), b.unwrap(), order
    else:
        return a.unwrap(), b.unwrap(order), order
