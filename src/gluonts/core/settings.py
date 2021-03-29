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
gluonts.core.settings
~~~~~~~~~~~~~~~~~~~~~

This modules offers a `Settings`-class, which allows to manage a global
context.

The idea is to support a form of dependency injection, where instead of passing
a concrete value along the call-chain, it is shared through the settings.

`gluonts.env` is a `Settings`.

Example::

    from gluonts.core.settings import Settings

    class MySettings(Settings):
        debug: bool = False

    settings = MySettings()

    def fn():
        if settings.debug:
            print("In debug mode.")

    with settings._let(debug=True):
        # this will print the message
        fn()

    # no message will be printed
    fn()

Another option is to inject the context to a function. This has the advantage,
that you can still manually pass values, but use the context as a fallback::

    @settings._inject("debug")
    def fn(debug):
        ...

    # this will use the value defined in the context
    fn()

    # but one can still set the value manually
    fn(False)


Value access is possible with both getitem (`setting["name"]`) and getattr
(`setting.name`). To avoid possible name-conflicts, all methods on `Settings`
use a leading underscore (e.g. `settings._let`). Consequently, keys are not
allowed to start with an underscore.

`Settings` contains a default-dictionary, which can be set to directly and is
used by `_declare`.

Additionally, it's possible to declare a type, which is checked using pydantic.
Whenever a new value is set, it is type-checked.

"""

import functools
import inspect
from typing import Any

import pydantic


class Dependency:
    def __init__(self, fn, dependencies):
        self.fn = fn
        self.dependencies = dependencies

    def resolve(self, env):
        kwargs = {key: env[key] for key in self.dependencies}
        return self.fn(**kwargs)


class _Config:
    arbitrary_types_allowed = True


class Settings:
    _cls_types: dict = {}
    _cls_deps: dict = {}

    def __init_subclass__(cls):
        cls._cls_types = {}

        for name, ty in cls.__annotations__.items():
            if ty == Dependency:
                cls._cls_deps[name] = getattr(cls, name)
            else:
                default = getattr(cls, name, ...)
                cls._cls_types[name] = ty, default

    def __init__(self, *args, **kwargs):
        # mapping of key to type, see `_declare` for more info on how this
        # works
        self._types = {}
        self._default = {}
        self._dependencies = {}
        self._context_count = 0

        # We essentially implement our own chainmap, managed by a list. New
        # entries appended to the right; thus, the chain acts as a stack. It is
        # ensured that there are always at least two entries in the chain:
        # A default, used to declare default values for any given key and a
        # base to guard from writing to the default through normal access.
        self._chain = [self._default, kwargs]

        # If sublcassed, `_cls_types` can contain declarations which we need to
        # execute.
        for key, (ty, default) in self._cls_types.items():
            self._declare(key, ty, default=default)

        # Same thing for dependencies.
        for name, fn in self._cls_deps.items():
            self._dependency(name, fn)

    def _reduce(self):
        """"""

        assert not self._context_count, "Cannot reduce within with-blocks."
        compact = {}

        for dct in self._chain[1:]:
            compact.update(dct)

        self._chain = [self._default, compact]

    def _already_declared(self, key):
        return key in self._types or key in self._dependencies

    def _declare(self, key, type=Any, *, default=...):
        assert not self._already_declared(
            key
        ), f"Attempt of overwriting already declared value {key}"

        # This is kinda hacky. For each key, we create a new pydantic model,
        # which contains just one definition, effectively, like this:
        #
        # class foo(pydantic.BaseModel):
        #     foo: type
        #
        # When we want to evaluate, we do this:
        #
        #    # given
        #    settings.foo = value
        #
        #    # becomes
        #    settings._types["foo"].parse_obj({"foo": value}).foo

        self._types[key] = pydantic.create_model(
            key, **{key: (type, ...)}, __config__=_Config
        )

        # we use our own default-handling, instead of relying on pydantic
        if default != ...:
            self._set_(self._default, key, default)

    def _dependency(self, name, fn):
        dependencies = list(inspect.signature(fn).parameters)
        for dependency in dependencies:
            assert self._already_declared(
                dependency
            ), f"`{name}` depends on `{dependency}`, which has not been declared yet."

        self._dependencies[name] = Dependency(fn, dependencies)

    def _get(self, key, default=None):
        """Like `dict.get`."""
        try:
            return self[key]
        except KeyError:
            return default

    def __getitem__(self, key):
        # Iterate all dicts, last to first, and return value as soon as one is
        # found.

        if key in self._dependencies:
            return self._dependencies[key].resolve(self)

        for dct in reversed(self._chain):
            try:
                return dct[key]
            except KeyError:
                pass
        raise KeyError(key)

    def __getattribute__(self, key):
        # We check the key, to check whether we want to acces our chainmap
        # or handle it as a normal attribute.
        if key.startswith("_"):
            return super().__getattribute__(key)
        else:
            return self[key]

    def _set_(self, dct, key, value):
        """Helper method to assign item to a given dictionary.

        Uses `_types` to type-check the value, before assigning.
        """

        assert key not in self._dependencies, "Can't override dependency."

        # If we have type-information, we apply the pydantic-model to the value
        model = self._types.get(key)
        if model is not None:
            value = getattr(model.parse_obj({key: value}), key)

        dct[key] = value

    def __setitem__(self, key, value):
        # Always assigns to the most recent dictionary in our chain.
        self._set_(self._chain[-1], key, value)

    def __setattr__(self, key, value):
        # Same check as in `__getattribute__`.
        if key.startswith("_"):
            super().__setattr__(key, value)
        else:
            self[key] = value

    def _push(self, **kwargs):
        """Add new entry to our chain-map.

        Values are type-checked.
        """
        self._chain.append({})
        # Since we want to type-check, we add the entries manually.
        for key, value in kwargs.items():
            self[key] = value

        return self

    def _pop(self):
        assert len(self._chain) > 2, "Can't pop initial setting."
        return self._chain.pop()

    def __repr__(self):
        inner = ", ".join(list(repr(dct) for dct in self._chain))
        return f"<Settings [{inner}]>"

    def _let(self, **kwargs) -> "_ScopedSettings":
        """Create a new context, where kwargs are added to the chain::

            with settings._let(foo=42):
                assert settings.foo = 42

        `_let` does not push a new context, but returns a `_ScopedSettings`
        object, that pushes the context, when entered through a
        `with`-statement.
        """
        return _ScopedSettings(self, kwargs)

    def _inject(self, *keys):
        """Dependency injection.

        This will inject values from settings if avaiable and not passed
        directly::

            @settings._inject("foo")
            def fn(foo=1):
                return foo

            # Since foo is not available in settings, the functions default
            # value is taken.
            assert fn() == 1

            with settings._let(foo=2):
                # Since foo is declared in the settings, it is used.
                assert fn() == 2

                # Directly passed values always take precedence.
                assert fn(3) == 3
        """

        def dec(fn):
            # We need the signature to be able to assemble the args later.
            sig = inspect.signature(fn)

            for key in keys:
                assert key in sig.parameters, f"Key {key} not in arguments."

            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                # arguments are always keyword params
                arguments = sig.bind_partial(*args, **kwargs).arguments

                setting_kwargs = {
                    key: self[key] for key in keys if key not in arguments
                }

                return fn(**arguments, **setting_kwargs)

            return wrapper

        return dec


class _ScopedSettings:
    def __init__(self, settings, kwargs):
        self.settings = settings
        self.kwargs = kwargs

    def __enter__(self):
        self.settings._context_count += 1
        return self.settings._push(**self.kwargs)

    def __exit__(self, *args):
        self.settings._context_count -= 1
        self.settings._pop()


def let(settings, **kwargs):
    "`let(settings, ...)` is the same as `settings._let(...)`."
    return settings._let(**kwargs)


def inject(settings, *args, **kwargs):
    "`inject(settings, ...)` is the same as `settings._inject(...)`."
    return settings._inject(*args, **kwargs)
