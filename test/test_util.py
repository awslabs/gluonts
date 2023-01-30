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

import tempfile
import tarfile
from pathlib import Path
from typing import Optional

import pytest

from gluonts.util import copy_with, will_extractall_into


def test_copy_with():
    class X:
        def __init__(self, value):
            self.value = value

    a = X(42)
    b = copy_with(a, value=99)

    assert a.value == 42
    assert b.value == 99


@pytest.mark.parametrize(
    "arcname, expect_failure",
    [
        (None, False),
        ("./file.txt", False),
        ("/a/../file.txt", False),
        ("/a/../../file.txt", True),
        ("../file.txt", True),
    ],
)
def test_will_extractall_into(arcname: Optional[str], expect_failure: bool):
    with tempfile.TemporaryDirectory() as tempdir:
        file_path = Path(tempdir) / "a" / "file.txt"
        file_path.parent.mkdir(parents=True)
        file_path.touch()

        with tarfile.open(Path(tempdir) / "archive.tar.gz", "w:gz") as tar:
            tar.add(file_path, arcname=arcname)

        if expect_failure:
            with pytest.raises(PermissionError):
                with tarfile.open(
                    Path(tempdir) / "archive.tar.gz", "r:gz"
                ) as tar:
                    will_extractall_into(tar, Path(tempdir) / "b")
        else:
            with tarfile.open(Path(tempdir) / "archive.tar.gz", "r:gz") as tar:
                will_extractall_into(tar, Path(tempdir) / "b")
