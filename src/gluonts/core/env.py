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

import functools
import inspect


class Context:
    def __init__(self, **kwargs):
        self._default = {}
        self.chain = [self._default, kwargs]

    def set_default(self, key, value, force=False):
        assert (
            force or key not in self._default
        ), "Attempt of overwriting default value"
        self._default[key] = value

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __getitem__(self, key):
        for dct in reversed(self.chain):
            try:
                return dct[key]
            except KeyError:
                pass
        raise KeyError(key)

    def __setitem__(self, key, value):
        self.chain[-1][key] = value

    def push(self, **kwargs):
        self.chain.append(kwargs)
        return self

    def pop(self):
        assert len(self.chain) > 2, "Can't pop initial context."
        return self.chain.pop()

    def __repr__(self):
        inner = ", ".join(list(repr(dct) for dct in self.chain))
        return f"<Context [{inner}]>"

    def __call__(self, **kwargs):
        return DelayedContext(self, kwargs)

    def bind(self, *keys, **values):
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
        return self.context.push(**self.kwargs)

    def __exit__(self, *args):
        return self.context.pop()


env = Context()
