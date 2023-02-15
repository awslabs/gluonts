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

import tarfile
from pathlib import Path


def will_extractall_into(tar: tarfile.TarFile, path: Path) -> None:
    """
    Check that the content of ``tar`` will be extracted within ``path``
    upon calling ``extractall``.

    Raise a ``PermissionError`` if not.
    """
    path = Path(path).resolve()

    for member in tar.getmembers():
        member_path = (path / member.name).resolve()

        try:
            member_path.relative_to(path)
        except ValueError:
            raise PermissionError(f"'{member.name}' extracts out of target.")


def safe_extractall(
    tar: tarfile.TarFile,
    path: Path = Path("."),
    members=None,
    *,
    numeric_owner=False,
):
    """
    Safe wrapper around ``TarFile.extractall`` that checks all destination
    files to be strictly within the given ``path``.
    """
    will_extractall_into(tar, path)
    tar.extractall(path, members, numeric_owner=numeric_owner)
