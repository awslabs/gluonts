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

import pytest
from pathlib import Path
from typing import Union

from gluonts.util import is_within_directory


@pytest.mark.parametrize(
    "directory, path, expected",
    [
        ("a/b/c", "a/b/c/d", True),
        ("./a", "./a/b", True),
        ("../a/b", "../a/b/c", True),
        ("a/./b/c", "a/b/./c/d", True),
        ("a/../b/c", "a/../b/c/d", True),
        ("a", "a", False),
        ("a/b/c", "a/b/c", False),
        ("./a/b/c", "./a/b/c", False),
        ("../a/b/c", "../a/b/c", False),
        ("a/b/c", "a/b/c/../d", False),
        ("./a", "./b", False),
        ("../a/b", "../b/c", False),
        ("a/../b/c", "a/b/../c/d", False),
    ],
)
def test_is_within_directory(
    directory: Union[str, Path], path: Union[str, Path], expected: bool
):
    assert is_within_directory(directory=directory, target=path) == expected
