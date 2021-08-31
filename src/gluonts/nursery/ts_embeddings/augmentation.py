import random

import numpy as np


class Jitter(object):
    """https://arxiv.org/pdf/1706.00527.pdf"""

    def __init__(self, p, sigma=0.03):
        self.p = p
        self.sigma = sigma

    def __call__(self, x):
        if random.random() < self.p:
            return x + np.random.normal(
                loc=0.0, scale=self.sigma, size=x.shape
            )
        else:
            return x


class Scaling(object):
    """https://arxiv.org/pdf/1706.00527.pdf"""

    def __init__(self, p, sigma=0.1):
        self.p = p
        self.sigma = sigma

    def __call__(self, x):
        if random.random() < self.p:
            factor = np.random.normal(
                loc=1.0, scale=self.sigma, size=(x.shape[0], x.shape[2])
            )
            return np.multiply(x, factor[:, np.newaxis, :])
        else:
            return x
