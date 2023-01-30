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

from gluonts.safe_extract import is_within_directory


@pytest.mark.parametrize(
    "directory, path, expected",
    [
        (Path("a"), Path("a"), True),
        (Path("a/b/c"), Path("a/b/c"), True),
        (Path("./a/b/c"), Path("./a/b/c"), True),
        (Path("../a/b/c"), Path("../a/b/c"), True),
        (Path("a/b/c"), Path("a/b/c/d"), True),
        (Path("./a"), Path("./a/b"), True),
        (Path("../a/b"), Path("../a/b/c"), True),
        (Path("a/b/c"), Path("a/b/c/../d"), False),
        (Path("./a"), Path("./b"), False),
        (Path("../a/b"), Path("../b/c"), False),
    ],
)
def test_is_within_directory(directory: Path, path: Path, expected: bool):
    assert is_within_directory(directory=directory, target=path) == expected
