import numpy as np


def get_level_issm_coeff(alpha):
    def g_t(t):
        return np.array([alpha[t % len(alpha)]])

    def a_t(t):
        return np.array([1.0])

    def F_t(t):
        return np.array([1.0])

    return a_t, F_t, g_t


def get_seasonality_issm_coeff(gamma):
    num_factors = len(gamma)
    eye = np.eye(num_factors)

    def g_t(t):
        return gamma[t % num_factors] * a_t(t)

    def a_t(t):
        return eye[:, t % num_factors]

    def F_t(t):
        return eye

    return a_t, F_t, g_t


def sample_issm(
    l0, a_t, g_t, sigma_t, T, t_initial=0, single_source_of_error=False
):
    target = np.zeros((T, 1))
    for idx_t, t in enumerate(range(t_initial, t_initial + T)):
        eps_t = np.random.normal(scale=sigma_t(t))
        if single_source_of_error:
            l0 = l0 + g_t(t) * eps_t
        else:
            l0 = l0 + g_t(t) * np.random.normal(scale=1.0)
        target[idx_t] = np.dot(a_t(t), l0) + eps_t
    return target, l0


def combine_funcs(a_t_list):
    def f_t(t):
        return np.concatenate([a_t(t) for a_t in a_t_list], axis=0)

    return f_t
