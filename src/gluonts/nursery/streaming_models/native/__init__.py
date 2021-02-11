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

import logging
from typing import Optional, Tuple

import numpy as np

try:
    from ._native import _ema
    from ._prcurve import (
        pyx_calculate_rewards,
        pyx_labels_to_ranges,
        pyx_precision_recall_curve,
        pyx_range_precision_recall,
        pyx_singleton_precision_recall,
    )
except:
    logging.error(
        "Could not load native extension directly. \r"
        "THIS SHOULD NOT HAPPEN IN THE PIPELINE OR IN PRODUCTION!!! \r"
        "Using pyximport"
    )
    import os
    from sys import platform

    import pyximport

    include_dirs = [
        np.get_include(),
    ]

    if platform == "darwin":
        # workaround for brew installed python on osx
        sdk_path = os.popen("xcrun --show-sdk-path").read().strip("\n")
        include_dirs.append(f"{sdk_path}/usr/include")

        cflags = " ".join(f"-I{path}" for path in include_dirs)
        os.environ["CFLAGS"] = cflags

    logging.info(f"Using includ dirs {include_dirs}")
    pyximport.install(setup_args={"include_dirs": include_dirs})
    from ._native import _ema
    from ._prcurve import (
        pyx_calculate_rewards,
        pyx_labels_to_ranges,
        pyx_precision_recall_curve,
        pyx_range_precision_recall,
        pyx_singleton_precision_recall,
    )


def ema(
    x: np.ndarray,
    alpha: float,
    minimum_value: float,
    initial_scale: float,
    state: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[float, float]]:
    """
    Exponential weighted moving average used for calculating a scale on a stream of data.

    We consider a value as "valid" if it is not NaN and has a magnitude larger than
    minimum_value. The scle is update when a value is "valid", until the first valid
    observation is passed, the scale returned is given by `initial_scale`.

    NOTE: The value x[i] at time i contributes to the scale[i] at the same time point.
    When used for forecasting models, make sure to take this into account to avoid leakage
    across time.


    Parameters
    ==========
    x
        1d array of float type
    alpha
        Smoothing factor used:
            w[i] = (1-alpha) * w[i-1] + alpha * x[i].
    minimum_value
        Minimum value that a data point has to have to be considered "valid"
    initial_scale
        Value of initial scale to use until the first "valid" data point.
    state
        State used to continue calculation.
        The state is a numpy array of length 4.

    Returns
    =======
    y
        result of averaging
    state
        state that can be used for continuing calculation
    """
    x = np.asarray(x)
    if state is not None:
        assert isinstance(state, np.ndarray)
        assert state.shape == (4,)
    else:
        state = np.zeros(4, dtype=x.dtype)
    assert x.dtype.kind == "f"
    assert len(x.shape) == 1
    return _ema(
        x=x,
        alpha=alpha,
        minimum_value=minimum_value,
        initial_scale=initial_scale,
        state=state,
    )
