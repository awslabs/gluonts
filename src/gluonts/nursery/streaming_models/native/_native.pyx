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
    cdef double m
    cdef double v
    cdef double a = alpha
    cdef double b = 1.0 - alpha
    cdef double min_val = minimum_value
    cdef double init_scale = initial_scale

    cdef double diff
    cdef double incre
    cdef double ratio

    s = state[0]
    w = state[1]
    m = state[2]
    v = state[3]

    cdef np.ndarray[floating, ndim=1] scale = np.zeros_like(x)
    cdef np.ndarray[floating, ndim=1] mean = np.zeros_like(x)
    cdef np.ndarray[floating, ndim=1] var = np.zeros_like(x)

    cdef Py_ssize_t i
    for i in range(x.shape[0]):
        if not isnan(x[i]) and abs(x[i]) >= min_val:
            s = (abs(x[i]) + b * w * s) / (1.0 + b * w)
            diff = x[i] - m
            incr = diff / (1.0 + b * w)
            m = m + incr 
            ratio = (b * w) / (1.0 + b * w)
            v = ratio * (v + diff * incr)
            w = 1.0 + b * w
        elif w == 0.0:
            m = initial_scale
            s = initial_scale
            v = initial_scale
        scale[i] = s
        mean[i] = m
        var[i] = v
    cdef np.ndarray[floating, ndim=1] new_state = np.zeros_like(state)
    new_state[0] = s
    new_state[1] = w
    new_state[2] = m
    new_state[3] = v
    return scale, mean, var, new_state
