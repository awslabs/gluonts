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

import copy
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar, Union

T = TypeVar("T")


def copy_with(obj: T, **kwargs) -> T:
    """Return copy of `obj` and update attributes of the copy using `kwargs`.

    ::

        @dataclass
        class MyClass:
            value: int

        a = MyClass(1)
        b = copy_with(a, value=2)

        assert a.value == 1
        assert b.value == 2

    """

    new_obj = copy.copy(obj)

    for name, value in kwargs.items():
        setattr(new_obj, name, value)

    return new_obj


if TYPE_CHECKING:
    lazy_property = property

else:

    def lazy_property(method):
        """Property that is lazily evaluated.

        This is the same as::

            @property
            @lru_cache(1)
            def my_property(self):
                ...

        In addition to be more concise, `lazy_property` also works with mypy,
        since it poses to be just `property` when type checked.

        This implementation follows the recipe from the `functools`
        documentation, and mimics `functools.cached_property` which was
        introduced in Python 3.8.
        """
        return property(lru_cache(1)(method))


def is_within_directory(
    directory: Union[str, Path], target: Union[str, Path]
) -> bool:
    """
    Check whether the ``target`` path is strictly within ``directory``.
    """
    abs_directory = Path(directory).absolute().resolve()
    abs_target = Path(target).absolute().resolve()
    return abs_directory in abs_target.parents


def safe_extract(
    tar, path: Path = Path("."), members=None, *, numeric_owner=False
):
    """
    Safe wrapper around ``TarFile.extractall`` that checks all destination
    files to be strictly within the given ``path``.
    """
    for member in tar.getmembers():
        member_path = path / member.name
        if not is_within_directory(path, member_path):
            raise Exception("Attempted Path Traversal in Tar File")

    tar.extractall(path, members, numeric_owner=numeric_owner)
