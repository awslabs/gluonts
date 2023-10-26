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
This modules contains pydantic imports, which are used throughout the codebase.
"""

try:
    import pydantic.v1 as pydantic

    PYDANTIC_V2 = True
except ModuleNotFoundError:
    PYDANTIC_V1 = False

if PYDANTIC_V2:
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
    import pydantic  # noqa
    from pydantic import (  # noqa
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
    from pydantic.error_wrappers import ValidationError, display_errors  # noqa
    from pydantic.utils import deep_update  # noqa
    from pydantic.dataclasses import dataclass  # noqa
