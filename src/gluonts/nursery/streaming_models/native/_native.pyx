#cython: language_level=3

from cython cimport floating
from libc.math cimport isnan

import numpy as np
cimport numpy as np

np.import_array()


def _ema(
        np.ndarray[floating, ndim=1] x,
        floating alpha,
        floating minimum_value,
        floating initial_scale,
        np.ndarray[floating, ndim=1] state,
):
    cdef double s
    cdef double w
    cdef double a = alpha
    cdef double b = 1.0 - alpha
    cdef double min_val = minimum_value
    cdef double init_scale = initial_scale

    s = state[0]
    w = state[1]

    cdef np.ndarray[floating, ndim=1] result = np.zeros_like(x)

    cdef Py_ssize_t i
    for i in range(x.shape[0]):
        if not isnan(x[i]) and abs(x[i]) >= min_val:
            s = (abs(x[i]) + b * w * s) / (1.0 + b * w)
            w = 1.0 + b * w
        elif w == 0.0:
            s = initial_scale
        result[i] = s
    cdef np.ndarray[floating, ndim=1] new_state = np.zeros_like(state)
    new_state[0] = s
    new_state[1] = w
    return result, new_state
