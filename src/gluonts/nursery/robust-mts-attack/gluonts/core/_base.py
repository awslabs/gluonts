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

from typing import Any


def fqname_for(obj: Any) -> str:
    """
    Returns the fully qualified name of ``obj``.

    Parameters
    ----------
    obj
        The class we are interested in.

    Returns
    -------
    str
        The fully qualified name of ``obj``.
    """

    if "<locals>" in obj.__qualname__:
        raise RuntimeError(
            "Can't get fully qualified name of locally defined object. "
            f"{obj.__qualname__}"
        )

    return f"{obj.__module__}.{obj.__qualname__}"
