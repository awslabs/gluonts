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

import os
import tarfile
from pathlib import Path
from typing import Optional, Set


def compress_directory(
    directory: Path, target: Path, include: Optional[Set[str]] = None
) -> None:
    """
    Compresses the provided directory into a single `.tar.gz` file.

    Args:
        directory: The directory to compress.
        target: The `.tar.gz` file where the compressed archive should be written.
        include: The filenames to include. If not provided, all files are included.
    """
    with target.open("wb+") as f:
        with tarfile.open(fileobj=f, mode="w:gz") as tar:
            for root, _, files in os.walk(directory):
                for file in files:
                    if include is not None and file not in include:
                        continue
                    name = os.path.join(root, file)
                    tar.add(name, arcname=os.path.relpath(name, directory))
