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

import json
from pathlib import Path
from typing import List, Optional
from .filters import Filter


def read_transform_write(
    file: Path,
    filters: Optional[List[Filter]] = None,
    source: Optional[Path] = None,
) -> None:
    """
    Reads the dataset from the provided path, applies the given transform and
    writes it back to the same file.

    Args:
        file: The path from where to read the data.
        filters: An optional list of filters to apply to modify the items in the datset.
        source: An optional source to read from. Defaults to the path to write to.
    """
    # Read
    data = []
    with (source or file).open("r") as f:
        for line in f:
            data.append(json.loads(line))

    # Filter
    for f in filters or []:
        data = f(data)

    # Write
    file.parent.mkdir(parents=True, exist_ok=True)
    with file.open("w") as f:
        content = "\n".join([json.dumps(d) for d in data])
        f.write(content + "\n")
