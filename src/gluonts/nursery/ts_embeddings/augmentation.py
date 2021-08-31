import random

import numpy as np
from scipy.interpolate import CubicSpline


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


class Rotation(object):
    def __init__(self, p, sigma=0.1):
        self.p = p

    def __call__(self, x):
        if random.random() < self.p:
            flip = np.random.choice([-1, 1], size=(x.shape[0], x.shape[2]))
            rotate_axis = np.arange(x.shape[2])
            np.random.shuffle(rotate_axis)

            return flip[:, np.newaxis, :] * x[:, :, rotate_axis]
        else:
            return x


class Permuation(object):
    def __init__(self, p, max_segments=5, seg_mode="equal"):
        self.p = p
        self.max_segments = max_segments
        self.seg_mode = seg_mode

    def __call__(self, x):
        if random.random() < self.p:
            orig_steps = np.arange(x.shape[1])

            num_segs = np.random.randint(
                1, self.max_segments, size=(x.shape[0])
            )

            ret = np.zeros_like(x)
            for i, pat in enumerate(x):
                if num_segs[i] > 1:
                    if self.seg_mode == "random":
                        split_points = np.random.choice(
                            x.shape[1] - 2, num_segs[i] - 1, replace=False
                        )
                        split_points.sort()
                        splits = np.split(orig_steps, split_points)
                    else:
                        splits = np.array_split(orig_steps, num_segs[i])
                    warp = np.concatenate(
                        np.random.permutation(splits)
                    ).ravel()
                    ret[i] = pat[warp]
                else:
                    ret[i] = pat
            return ret
        else:
            return x


class MagnitudeWarp(object):
    def __init__(self, p, sigma=0.2, knot=4):
        self.p = p
        self.sigma = sigma
        self.knot = knot

    def __call(self, x):
        if random.random() < self.p:
            orig_steps = np.arange(x.shape[1])

            random_warps = np.random.normal(
                loc=1.0,
                scale=self.sigma,
                size=(x.shape[0], self.knot + 2, x.shape[2]),
            )
            warp_steps = (
                np.ones((x.shape[2], 1))
                * (np.linspace(0, x.shape[1] - 1.0, num=self.knot + 2))
            ).T
            ret = np.zeros_like(x)
            for i, pat in enumerate(x):
                warper = np.array(
                    [
                        CubicSpline(
                            warp_steps[:, dim], random_warps[i, :, dim]
                        )(orig_steps)
                        for dim in range(x.shape[2])
                    ]
                ).T
                ret[i] = pat * warper

            return ret
        else:
            return x


class TimeWrap(object):
    def __init__(self, p, sigma=0.2, knot=4):
        self.p = p
        self.sigma = sigma
        self.knot = knot

    def __call__(self, x):
        if random.random() < self.p:
            orig_steps = np.arange(x.shape[1])

            random_warps = np.random.normal(
                loc=1.0,
                scale=self.sigma,
                size=(x.shape[0], self.knot + 2, x.shape[2]),
            )
            warp_steps = (
                np.ones((x.shape[2], 1))
                * (np.linspace(0, x.shape[1] - 1.0, num=self.knot + 2))
            ).T

            ret = np.zeros_like(x)
            for i, pat in enumerate(x):
                for dim in range(x.shape[2]):
                    time_warp = CubicSpline(
                        warp_steps[:, dim],
                        warp_steps[:, dim] * random_warps[i, :, dim],
                    )(orig_steps)
                    scale = (x.shape[1] - 1) / time_warp[-1]
                    ret[i, :, dim] = np.interp(
                        orig_steps,
                        np.clip(scale * time_warp, 0, x.shape[1] - 1),
                        pat[:, dim],
                    ).T
            return ret
        else:
            return x
