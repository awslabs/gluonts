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


class BaseConfig:
    arbitrary_types_allowed = True


def checked(fn):
    fn_params = inspect.signature(fn).parameters
    fn_fields = {
        param.name: (get_param_type(param), get_param_default(param),)
        for param in fn_params.values()
        if param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
    }

    Model = pydantic.create_model(
        f"{fn.__name__}Model", __config__=BaseConfig, **fn_fields,
    )

    @functools.wraps(fn)
    def fn_wrapper(*args, **kwargs):
        nmargs = {name: arg for name, arg in zip(fn_params, args)}

        try:
            model = Model(**nmargs, **kwargs)
        except pydantic.ValidationError as err:
            errors = err.errors()

            details = "\n".join(
                [f'\t{error["loc"][0]}: {error["msg"]}' for error in errors]
            )

            raise TypeError(
                f'Cannot call "{fn.__qualname__}":\n{details}'
            ) from err

        return fn(**model.dict())

    return fn_wrapper
