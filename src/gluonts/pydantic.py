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


"""This modules contains pydantic imports, which are used throughout the codebase."""

from pydantic import __version__

if __version__.startswith("2"):
    import pydantic.v1 as pydantic
    from pydantic.v1 import (
        BaseConfig,
        BaseModel,
        create_model,
        root_validator,
        PositiveInt,
        PrivateAttr,
        Field,
        parse_obj_as,
        PositiveFloat,
        BaseSettings,
    )
    from pydantic.v1.error_wrappers import ValidationError, display_errors
    from pydantic.v1.utils import deep_update
    from pydantic.v1.dataclasses import dataclass
else:
    import pydantic
    from pydantic import (
        BaseConfig,
        BaseModel,
        create_model,
        root_validator,
        PositiveInt,
        PrivateAttr,
        Field,
        parse_obj_as,
        PositiveFloat,
        BaseSettings,
    )
    from pydantic.error_wrappers import ValidationError, display_errors
    from pydantic.utils import deep_update
    from pydantic.dataclasses import dataclass


__all__ = [
    "BaseConfig",
    "BaseModel",
    "BaseSettings",
    "Field",
    "PositiveFloat",
    "PositiveInt",
    "PrivateAttr",
    "ValidationError",
    "__version__",
    "create_model",
    "dataclass",
    "deep_update",
    "display_errors",
    "parse_obj_as",
    "pydantic",
    "root_validator",
]
