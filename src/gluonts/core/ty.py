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
import itertools
from typing import Any, Optional

import pydantic


def get_param_type(param):
    if param.annotation == inspect.Parameter.empty:
        return Any

    return param.annotation


def get_param_default(param):
    if param.default == inspect.Parameter.empty:
        return ...

    return param.default


def checked(fn):
    assert not inspect.isclass(fn)
    fn_params = inspect.signature(fn).parameters

    has_var_args = any(
        param.kind in [param.VAR_KEYWORD, param.VAR_POSITIONAL]
        for param in fn_params.values()
    )

    fn_params = {
        key: param
        for key, param in fn_params.items()
        if param.kind not in [param.VAR_KEYWORD, param.VAR_POSITIONAL]
    }

    class Config:
        arbitrary_types_allowed = True
        extra = "ignore" if has_var_args else "forbid"

    fn_fields = {
        param.name: (
            get_param_type(param),
            get_param_default(param),
        )
        for param in fn_params.values()
    }

    Model = pydantic.create_model(
        f"{fn.__name__}Model",
        __config__=Config,
        **fn_fields,
    )

    def checked_args(*args, **kwargs):
        nmargs = {
            name: arg
            for name, arg in zip(fn_params, args)
            if name not in kwargs
        }

        try:
            model = Model(**{**kwargs, **nmargs})
        except pydantic.ValidationError as err:
            errors = err.errors()

            details = "\n".join(
                [f'\t{error["loc"][0]}: {error["msg"]}' for error in errors]
            )

            raise TypeError(
                f'Cannot call "{fn.__qualname__}":\n{details}'
            ) from err

        typed_kwargs = {**kwargs, **model.dict()}
        typed_args = [typed_kwargs.pop(name) for name in nmargs]
        typed_args += args[len(typed_args) :]

        return typed_args, typed_kwargs

    @functools.wraps(fn)
    def fn_wrapper(*args, **kwargs):
        typed_args, typed_kwargs = checked_args(*args, **kwargs)
        return fn(*typed_args, **typed_kwargs)

    fn_wrapper.__checked__ = checked_args

    return fn_wrapper
