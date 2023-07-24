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

from .core.settings import Settings


# see: https://wiki.archlinux.org/title/XDG_Base_Directory
if "GLUONTS_DATA" in os.environ:
    data_path = Path(os.environ["GLUONTS_DATA"]).expanduser()
else:
    data_path = (
        Path(os.environ.get("XDG_DATA_HOME", "~/.local/share")).expanduser()
        / "gluonts"
    )


class Environment(Settings):
    # Maximum number of times a transformation can receive an input without
    # returning an output. This parameter is intended to catch infinite loops
    # or inefficiencies, when transformations never or rarely return
    # something.
    max_idle_transforms: int = os.environ.get(  # type: ignore
        "GLUONTS_MAX_IDLE_TRANSFORMS", "100"
    )

    # we want to be able to disable TQDM, for example when running in sagemaker
    use_tqdm: bool = True

    data_path: Path = data_path

    def get_data_path(self, create: bool = True) -> Path:
        path = self.data_path
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)

        return path


env = Environment()
