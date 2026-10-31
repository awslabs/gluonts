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


import numpy as np
from scipy.interpolate import CubicSpline

import torch
import torch.nn as nn


class RandomApply(nn.Module):
    def __init__(self, transforms, p=0.5):
        super().__init__()
        self.transforms = nn.ModuleList(transforms)
        self.p = p

    def forward(self, x):
        if self.p < torch.rand(1):
            return x

        x_time_first = x.transpose(2, 1)
        for t in self.transforms:
            x_time_first = t(x_time_first)

        return x_time_first.transpose(1, 2)


class Jitter(nn.Module):
    """https://arxiv.org/pdf/1706.00527.pdf"""

    def __init__(self, p, sigma=0.03):
        super().__init__()
        self.p = p
        self.sigma = sigma

    def forward(self, x):
        if self.p < torch.rand(1):
            return x

        return x + torch.normal(
            mean=0.0, std=self.sigma, size=x.shape, device=x.device
        )


class Scaling(nn.Module):
    """https://arxiv.org/pdf/1706.00527.pdf"""

    def __init__(self, p, sigma=0.1):
        super().__init__()
        self.p = p
        self.sigma = sigma

    def forward(self, x):
        if self.p < torch.rand(1):
            return x
        factor = torch.normal(
            mean=1.0,
            std=self.sigma,
            size=(x.shape[0], 1, x.shape[2]),
            device=x.device,
        )
        return x * factor


class Rotation(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.p < torch.rand(1):
            return x

        flip_index = torch.multinomial(
            torch.tensor([0.5, 0.5], dtype=x.dtype, device=x.device),
            num_samples=x.shape[0] * x.shape[2],
            replacement=True,
        )

        ones = torch.ones((x.shape[0] * x.shape[2]), device=x.device)
        flip = torch.where(flip_index == 0, -ones, ones)

        rotate_axis = np.arange(x.shape[2])
        np.random.shuffle(rotate_axis)

        return flip.reshape(x.shape[0], 1, x.shape[2]) * x[:, :, rotate_axis]


class Permuation(nn.Module):
    def __init__(self, p, max_segments=5, seg_mode="equal"):
        super().__init__()
        self.p = p
        self.max_segments = max_segments
        self.seg_mode = seg_mode

    def forward(self, x):
        if self.p < torch.rand(1):
            return x

        orig_steps = np.arange(x.shape[1])

        num_segs = np.random.randint(1, self.max_segments, size=(x.shape[0]))

        ret = torch.zeros_like(x)
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
                warp = np.concatenate(np.random.permutation(splits)).ravel()
                ret[i] = pat[warp]
            else:
                ret[i] = pat
        return ret


class MagnitudeWarp(nn.Module):
    def __init__(self, p, sigma=0.2, knot=4):
        super().__init__()
        self.p = p
        self.sigma = sigma
        self.knot = knot

    def forward(self, x):
        if self.p < torch.rand(1):
            return x

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

        ret = torch.zeros_like(x)
        for i, pat in enumerate(x):
            warper = np.array(
                [
                    CubicSpline(warp_steps[:, dim], random_warps[i, :, dim])(
                        orig_steps
                    )
                    for dim in range(x.shape[2])
                ]
            ).T

            ret[i] = pat * torch.from_numpy(warper).float().to(x.device)

        return ret


class TimeWrap(nn.Module):
    def __init__(self, p, sigma=0.2, knot=4):
        super().__init__()
        self.p = p
        self.sigma = sigma
        self.knot = knot

    def forward(self, x):
        if self.p < torch.rand(1):
            return x

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

        ret = torch.zeros_like(x)
        for i, pat in enumerate(x):
            for dim in range(x.shape[2]):
                time_warp = CubicSpline(
                    warp_steps[:, dim],
                    warp_steps[:, dim] * random_warps[i, :, dim],
                )(orig_steps)
                scale = (x.shape[1] - 1) / time_warp[-1]
                wrap = np.interp(
                    orig_steps,
                    np.clip(scale * time_warp, 0, x.shape[1] - 1),
                    pat[:, dim].cpu().numpy(),
                ).T
                ret[i, :, dim] = torch.from_numpy(wrap).float().to(x.device)

        return ret


class WindowSlice(nn.Module):
    """https://halshs.archives-ouvertes.fr/halshs-01357973/document"""

    def __init__(self, p, reduce_ratio=0.9):
        super().__init__()
        self.p = p
        self.reduce_ratio = reduce_ratio

    def forward(self, x):
        if self.p < torch.rand(1):
            return x

        target_len = np.ceil(self.reduce_ratio * x.shape[1]).astype(int)
        if target_len >= x.shape[1]:
            return x
        starts = np.random.randint(
            low=0, high=x.shape[1] - target_len, size=(x.shape[0])
        ).astype(int)
        ends = (target_len + starts).astype(int)

        ret = torch.zeros_like(x)
        for i, pat in enumerate(x):
            for dim in range(x.shape[2]):
                warp = np.interp(
                    np.linspace(0, target_len, num=x.shape[1]),
                    np.arange(target_len),
                    pat[starts[i] : ends[i], dim].cpu().numpy(),
                ).T
                ret[i, :, dim] = torch.from_numpy(warp).float().to(x.device)
        return ret


class WindowWarp(nn.Module):
    """https://halshs.archives-ouvertes.fr/halshs-01357973/document"""

    def __init__(self, p, window_ratio=0.1, scales=[0.5, 2.0]):
        super().__init__()
        self.p = p
        self.window_ratio = window_ratio
        self.scales = scales

    def forward(self, x):
        if self.p < torch.rand(1):
            return x

        warp_scales = np.random.choice(self.scales, x.shape[0])
        warp_size = np.ceil(self.window_ratio * x.shape[1]).astype(int)
        window_steps = np.arange(warp_size)

        window_starts = np.random.randint(
            low=1, high=x.shape[1] - warp_size - 1, size=(x.shape[0])
        ).astype(int)
        window_ends = (window_starts + warp_size).astype(int)

        ret = torch.zeros_like(x)
        for i, pat in enumerate(x):
            for dim in range(x.shape[2]):
                start_seg = pat[: window_starts[i], dim].cpu().numpy()
                window_seg = np.interp(
                    np.linspace(
                        0,
                        warp_size - 1,
                        num=int(warp_size * warp_scales[i]),
                    ),
                    window_steps,
                    pat[window_starts[i] : window_ends[i], dim].cpu().numpy(),
                )
                end_seg = pat[window_ends[i] :, dim].cpu().numpy()
                warped = np.concatenate((start_seg, window_seg, end_seg))
                warp = np.interp(
                    np.arange(x.shape[1]),
                    np.linspace(0, x.shape[1] - 1.0, num=warped.size),
                    warped,
                ).T
                ret[i, :, dim] = torch.from_numpy(warp).float().to(x.device)
        return ret
