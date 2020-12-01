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
import signal
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np


def pad_to_size(
    x: np.array, size: int, axis: int = 0, is_right_pad: bool = True
):
    """Pads `xs` with 0 on the right (default) on the specified axis, which is the first axis by default."""
    pad_length = size - x.shape[axis]
    if pad_length <= 0:
        return x

    pad_width = [(0, 0)] * x.ndim
    right_pad = (0, pad_length)
    pad_width[axis] = right_pad if is_right_pad else right_pad[::-1]
    return np.pad(x, mode="constant", pad_width=pad_width)


class Timer:
    """Context manager for measuring the time of enclosed code fragments."""

    def __enter__(self):
        self.start = time.perf_counter()
        self.interval = None
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.interval = self.end - self.start


class SignalHandler:
    """
    A context manager that attaches a set of signal handlers within its scope.

    Parameters
    ----------
    handlers_map
        A dictionary mapping signal numbers to associated signal handlers to
        be attached within the scope of the enclosing `SignalHandler` instance.
    """

    Callback = Optional[Callable[[int, Any], None]]

    def __init__(self, handlers_map: Dict[int, Callback]) -> None:
        self.handlers_map = handlers_map

    def __enter__(self):
        self.default_handlers = {
            s: signal.signal(s, h) for s, h in self.handlers_map.items()
        }
        return self

    def __exit__(self, *args):
        for s, h in self.default_handlers.items():
            signal.signal(s, h)


def maybe_len(obj) -> Optional[int]:
    try:
        return len(obj)
    except NotImplementedError:
        return None


def get_download_path() -> Path:
    """

    Returns
    -------
    Path
        default path to download datasets or models of gluon-ts.
        The path is either $MXNET_HOME if the environment variable is defined or
        /home/username/.mxnet/gluon-ts/
    """
    return Path(
        os.environ.get("MXNET_HOME", str(Path.home() / ".mxnet" / "gluon-ts"))
    )


def map_dct_values(fn: Callable, dct: dict) -> dict:
    """Maps `fn` over a dicts values."""
    return {key: fn(value) for key, value in dct.items()}
