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
from .core.settings import Settings


class Environment(Settings):
    # Maximum number of times a transformation can receive an input without
    # returning an output. This parameter is intended to catch infinite loops
    # or inefficiencies, when transformations never or rarely return
    # something.
    max_idle_transforms: int = os.environ.get(
        "GLUONTS_MAX_IDLE_TRANSFORMS", "100"
    )

    # we want to be able to disable TQDM, for example when running in sagemaker
    use_tqdm: bool = True


env = Environment()
