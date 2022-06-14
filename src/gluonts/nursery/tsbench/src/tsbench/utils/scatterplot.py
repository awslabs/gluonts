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

from dataclasses import dataclass
from typing import List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tsbench.recommender.utils import pareto_efficiency_mask


@dataclass
class Dimension:
    """
    A dimension describes a combination of metric name and its display name.
    """

    metric: str
    displayName: str


@dataclass
class DataEntry:
    """
    A data entry describes a set of datapoints to be displayed and how they
    should be displayed.
    """

    data: pd.DataFrame
    color: str
    marker: str
    size: int
    edgecolor: Optional[str] = None
    edgewidth: Optional[int] = None
    name: Optional[str] = None
    alpha: float = 1


def plot_scatter_matrix(  # pylint: disable=too-many-statements
    data_entries: List[DataEntry],
    dimensions: List[Dimension],
    title: Optional[str] = None,
    savefig: Optional[str] = None,
    position_legend: bool = True,
    size: Tuple[int, int] = (4, 3),
    tick_fontsize: Optional[int] = None,
    axis_fontsize: Optional[int] = None,
    legend_fontsize: Optional[int] = None,
    legend_ncol: int = 1,
    ylog: bool = True,
    xmul: float = 1,
    plot_pareto_front: bool = False,
) -> None:
    """
    Creates a set of plots that visualize the data entries along the provided
    dimensions.
    """
    n = len(dimensions)
    fig, axes = plt.subplots(
        n - 1, n - 1, figsize=((n - 1) * size[0], (n - 1) * size[1]), dpi=150
    )
    if title is not None:
        fig.suptitle(title)

    for i in range(n - 1):
        for j in range(n - 1):
            ax = axes[i, j] if n > 2 else axes
            if j > i:
                ax.axis("off")
                continue

            xs = []
            ys = []
            for entry in data_entries:
                ax.scatter(
                    entry.data[dimensions[j].metric] * xmul,
                    entry.data[dimensions[i + 1].metric],
                    marker=entry.marker,  # type: ignore
                    s=entry.size,
                    color=entry.color,
                    edgecolors=entry.edgecolor,
                    linewidths=entry.edgewidth,
                    label=entry.name,
                    alpha=entry.alpha,
                )
                xs.extend((entry.data[dimensions[j].metric] * xmul).to_list())
                ys.extend(entry.data[dimensions[i + 1].metric].to_list())

            ax.set_xscale("log")
            if ylog:
                ax.set_yscale("log")

            if plot_pareto_front:
                ylim = ax.get_ylim()
                xlim = ax.get_xlim()
                xs = np.array(xs)
                ys = np.array(ys)
                all_points = np.stack([xs, ys], axis=1)
                mask = pareto_efficiency_mask(all_points)
                pareto_front = np.array(
                    sorted(all_points[mask], key=lambda r: -r[1])
                )
                pareto_front_x = pareto_front[:, 0].tolist()
                pareto_front_y = pareto_front[:, 1].tolist()
                ax.step(
                    [pareto_front_x[0]] + pareto_front_x + [1e10],
                    [1e10] + pareto_front_y + [pareto_front_y[-1]],
                    where="post",
                    color="black",
                    linewidth=1.5,
                    label="Pareto Front" if i == 0 else None,
                    zorder=-1000,
                )
                ax.set_ylim(ylim)
                ax.set_xlim(xlim)

            if j == 0:
                ax.set_ylabel(
                    dimensions[i + 1].displayName, fontsize=axis_fontsize
                )
            if tick_fontsize is not None:
                for tick in ax.yaxis.get_major_ticks():
                    tick.label.set_fontsize(tick_fontsize)

            if i == n - 2:
                ax.set_xlabel(
                    dimensions[j].displayName, fontsize=axis_fontsize
                )
            if tick_fontsize is not None:
                for tick in ax.xaxis.get_major_ticks():
                    tick.label.set_fontsize(tick_fontsize)

            handles, labels = ax.get_legend_handles_labels()

    legend_anchor = ax if n == 2 else fig  # type: ignore
    if position_legend:
        legend_anchor.legend(
            handles,  # type: ignore
            labels,  # type: ignore
            loc="upper right",
            bbox_to_anchor=(0, 0, 0.9, 0.9),
            fontsize=legend_fontsize,
            ncol=legend_ncol,
        )
    else:
        legend_anchor.legend(
            handles,  # type: ignore
            labels,  # type: ignore
            fontsize=legend_fontsize,
            ncol=legend_ncol,
        )
    plt.tight_layout()
    if savefig is not None:
        fig.patch.set_facecolor("white")
        plt.savefig(savefig)
    plt.show()
