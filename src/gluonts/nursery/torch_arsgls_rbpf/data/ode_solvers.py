import numpy as np


def rungekutta4(state: np.ndarray, t: float, dt: float, f: callable):
    k1 = dt * f(state=state, t=t)
    k2 = dt * f(state=state + 0.5 * k1, t=t + 0.5 * dt)
    k3 = dt * f(state=state + 0.5 * k2, t=t + 0.5 * dt)
    k4 = dt * f(state=state + k3, t=t + dt)

    next_state = state + (k1 + 2.0 * (k2 + k3) + k4) / 6.0
    next_t = t + dt
    return next_state, next_t
