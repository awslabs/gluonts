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

"""
gluonts.core.context
~~~~~~~~~~~~~~~~~~~~

This modules offers a `Context`-class, which allows to manage a global context.

The idea is to support a form of dependency injection, where instead of passing
a concrete value along the call-chain, it is shared through the context.

`gluonts.env` is such a context and is used to manage global settings, such as
the number of workers in multiprocessing.

Example::
    from gluonts.core.context import Context

    class MyContext(Context):
        debug: bool = False

    def fn():
        if ctx.debug:
            print("In debug mode.")

    # we use `_let` instead of `let` to avoid possible name-collisions
    # so you can do `ctx._let(let=...)`
    with ctx._let(debug=True):
        # this will print the message
        fn()

    # no message will be printed
    fn()

Another option is to inject the context to a function. This has the advantage,
that you can still manually pass values, but use the context as a fallback::


    @ctx._inject("debug")
    def fn(debug):
        ...

    # this will use the value defined in the context
    fn()

    # but one can still set the value manually
    fn(False)
"""

import functools
import inspect
from typing import Any

import pydantic


class _Config:
    arbitrary_types_allowed = True


class Context:
    _cls_types = {}

    def __init_subclass__(cls):
        cls._cls_types = {}

        for name, ty in cls.__annotations__.items():
            default = getattr(cls, name, ...)
            cls._cls_types[name] = ty, default

    def __init__(self, **kwargs):
        self._default = {}
        self._types = {}
        self._chain = [self._default, kwargs]

        for key, (ty, default) in self._cls_types.items():
            self._declare(key, ty, default=default)

    def _declare(self, key, type=Any, *, default=..., force=False):
        assert (
            force or key not in self._types
        ), f"Attempt of overwriting already declared value {key}"

        self._types[key] = pydantic.create_model(
            key, **{key: (type, ...)}, __config__=_Config
        )

        if default != ...:
            self._set_(self._default, key, default)

    def _get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __getitem__(self, key):
        for dct in reversed(self._chain):
            try:
                return dct[key]
            except KeyError:
                pass
        raise KeyError(key)

    def __getattribute__(self, key):
        if key.startswith("_"):
            return super().__getattribute__(key)
        else:
            return self[key]

    def _set_(self, dct, key, value):
        model = self._types.get(key)
        if model is not None:
            value = getattr(model.parse_obj({key: value}), key)

        dct[key] = value

    def __setitem__(self, key, value):
        self._set_(self._chain[-1], key, value)

    def __setattr__(self, key, value):
        if key.startswith("_"):
            self.__dict__[key] = value
        else:
            self[key] = value

    def _push(self, **kwargs):
        self._chain.append({})
        for key, value in kwargs.items():
            self[key] = value
        return self

    def _pop(self):
        assert len(self._chain) > 2, "Can't pop initial context."
        return self._chain.pop()

    def __repr__(self):
        inner = ", ".join(list(repr(dct) for dct in self._chain))
        return f"<Context [{inner}]>"

    def _let(self, **kwargs):
        return DelayedContext(self, kwargs)

    def _inject(self, *keys, **values):
        def dec(fn):
            sig = inspect.signature(fn)

            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                env_kwargs = {}
                for key in keys:
                    try:
                        env_kwargs[key] = self[key]
                    except KeyError:
                        pass
                env_kwargs.update(
                    {
                        key: self.get(key, default)
                        for key, default in values.items()
                    }
                )
                return fn(
                    **{
                        **env_kwargs,
                        **sig.bind_partial(*args, **kwargs).arguments,
                    }
                )

            return wrapper

        return dec


class DelayedContext:
    def __init__(self, context, kwargs):
        self.context = context
        self.kwargs = kwargs

    def __enter__(self):
        return self.context._push(**self.kwargs)

    def __exit__(self, *args):
        self.context._pop()


def let(context, **kwargs):
    "`let(ctx, ...)` is the same as `ctx._let(...)`."
    return context._let(**kwargs)


def inject(context, *args, **kwargs):
    "`inject(ctx, ...)` is the same as `ctx._inject(...)`."
    return context._inject(*args, **kwargs)
