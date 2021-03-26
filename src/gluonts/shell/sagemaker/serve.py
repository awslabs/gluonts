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
from pathlib import Path


class ServePaths:
    def __init__(self, base: Path = Path("/opt/ml")) -> None:
        self.base = base.expanduser().resolve()
        self.model = self.base / "model"
        self.output = self.base / "output"

        self.model.mkdir(parents=True, exist_ok=True)
        self.output.mkdir(parents=True, exist_ok=True)


class ServeEnv:
    def __init__(self, path: Path = Path("/opt/ml")) -> None:
        self.path = ServePaths(path)

        self.sagemaker_batch = (
            os.environ.get("SAGEMAKER_BATCH", "false") == "true"
        )
