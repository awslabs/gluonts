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

# Standard library imports
import functools
import sys

# Third-party imports
from tqdm import tqdm as _tqdm

# TODO: when we have upgraded this will give notebook progress bars
# from tqdm.auto import tqdm as _tqdm


USE_TQDM = True


@functools.wraps(_tqdm)
def tqdm(it, *args, **kwargs):
    # we want to be able to disable TQDM, for example when running in sagemaker
    if not USE_TQDM:
        return it

    kwargs = kwargs.copy()
    if not sys.stdout.isatty():
        kwargs.update(mininterval=10.0)

    return _tqdm(it, *args, **kwargs)
