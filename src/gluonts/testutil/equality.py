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

import pandas as pd
import numpy as np


def assert_recursively_equal(obj_a, obj_b, equal_nan=True):
    """
    Asserts that two objects are equal, recursively.

    This is based on :func:`assert_recursively_close`, and accepts the
    same arguments, except that tolerances are set to zero.

    Parameters:
    -----------
    obj_a
    obj_b
        Objects to compare.
    equal_nan
        Indicates whether or not numpy.nan values should be considered equal.
    """
    _assert_recursively_close(
        obj_a, obj_b, location="", rtol=0, atol=0, equal_nan=equal_nan
    )


def assert_recursively_close(
    obj_a, obj_b, rtol=1e-05, atol=1e-08, equal_nan=True
):
    """
    Asserts that two objects are "close" to each other, recursively.

    Strings or ints are close iff they are equal; floats or numpy arrays
    are defined close according to the numpy.isclose and numpy.allclose
    functions, respectively. Lists are close if all of their items are close.
    Dicts are close if they have the same keys, and elements corresponding
    to the same key are close.

    Parameters:
    -----------
    obj_a
    obj_b
        Objects to compare.
    rtol
    atol
        Relative and absolute tolerance for float comparison; see docs for
        numpy.isclose.
    equal_nan
        Indicates whether or not numpy.nan values should be considered equal.
    """
    _assert_recursively_close(
        obj_a, obj_b, location="", rtol=rtol, atol=atol, equal_nan=equal_nan
    )


def _assert_recursively_close(obj_a, obj_b, location, *args, **kwargs):
    assert type(obj_a) == type(
        obj_b
    ), f"types don't match (at {location}) {type(obj_a)} != {type(obj_b)}"
    if isinstance(obj_a, (str, int)):
        assert obj_a == obj_b
    elif isinstance(obj_a, float):
        assert np.isclose(obj_a, obj_b, *args, **kwargs)
    elif isinstance(obj_a, list):
        assert len(obj_a) == len(obj_b), f"lengths don't match (at {location})"
        for i, (element_a, element_b) in enumerate(zip(obj_a, obj_b)):
            _assert_recursively_close(
                element_a,
                element_b,
                location=f"{location}.{i}",
                *args,
                **kwargs,
            )
    elif isinstance(obj_a, dict):
        assert (
            obj_a.keys() == obj_b.keys()
        ), f"keys don't match (at {location})"
        for k in obj_a:
            _assert_recursively_close(
                obj_a[k], obj_b[k], location=f"{location}.{k}", *args, **kwargs
            )
    elif isinstance(obj_a, np.ndarray):
        assert (
            obj_a.dtype == obj_b.dtype
        ), f"numpy arrays have different dtype (at {location})"
        assert np.allclose(
            obj_a, obj_b, *args, **kwargs
        ), f"numpy arrays are not close enough (at {location})"
    elif isinstance(obj_a, pd.Period):
        assert obj_a == obj_b
    elif obj_a is None:
        assert obj_b is None
    else:
        raise TypeError(f"unsupported type {type(obj_a)}")
