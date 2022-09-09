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

from inspect import signature
from typing import Callable, Dict, Any


def matching_arguments(fn: Callable, args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract valid keyword arguments for a Callable from a dictionary.

    Parameters
    ----------
    fn
        Callable for which matching arguments are required.
    args
        A dictionary containing keyword arguments.

    Returns
    -------
    Dict
        A dictionary containing valid keyword arguments.

    """
    return {p: args[p] for p in signature(fn).parameters.keys() & args.keys()}
