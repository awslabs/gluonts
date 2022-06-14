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

from typing import Tuple
import numpy as np
import numpy.typing as npt
import pygmo
import scipy.spatial as spt


def hypervolume(points: npt.NDArray[np.float32], box: float = 1) -> float:
    """
    Computes the hypervolume assuming that all values were previously
    normalized to lie in the range (0, 1).

    Args:
        point: Array of shape [N, D] (N: number of points, D: dimensionality) containing the points
            of the Pareto front.
        box: The coordinate of the box's edge. By default, this is located at [1, ..., 1] with
            the appropriate dimensionality.

    Returns:
        The hypervolume.
    """
    dim = points.shape[1]
    ref = np.ones(dim) * box
    return pygmo.hypervolume(points).compute(ref) / (box**dim)  # type: ignore


def maximum_spread(solution: npt.NDArray[np.float32]) -> float:
    """
    Computes the maximum spread of the solution's Pareto front.

    Args:
        solution: Array of shape [N, D] (N: number of points, D: dimensionality) containing the
            points of the solution set.

    Returns:
        The modified maximum spread of the solution.
    """
    solution_min = np.min(solution, axis=0)
    solution_max = np.max(solution, axis=0)
    return np.linalg.norm(solution_max - solution_min)


def pure_diversity(solution: npt.NDArray[np.float32]) -> float:
    """
    Computes the pure diversity of the solutions as described in "Diversity
    Assessment in Many-Objective Optimization" (Wang et al., 2017).

    Args:
        solution: Array of shape [N, D] (N: number of points, D: dimensionality) containing the
            points of the solution set.

    Returns:
        The modified maximum spread of the solution.
    """
    distances = spt.distance_matrix(solution, solution, p=2)
    np.fill_diagonal(distances, float("inf"))

    return _compute_pd(distances)


def _compute_pd(distances: npt.NDArray[np.float32]) -> float:
    pd = 0

    n = distances.shape[0]
    adjacency = np.zeros((n, n), dtype=bool)
    for _ in range(n - 1):
        i, j, dist = _get_ij_dist(distances)

        while _is_connected(adjacency, i, j):
            if distances[i, j] != -1:
                distances[i, j] = float("inf")
            if distances[j, i] != -1:
                distances[j, i] = float("inf")
            i, j, dist = _get_ij_dist(distances)

        adjacency[i, j] = True
        adjacency[j, i] = True
        pd += dist
        if distances[j, i] == -1:
            distances[j, i] = float("inf")
        distances[i] = -1

    return pd


def _get_ij_dist(distances: npt.NDArray[np.float32]) -> Tuple[int, int, float]:
    min_distances = distances.min(1)
    min_distance_indices = distances.argmin(1)
    i = min_distances.argmax()
    j = min_distance_indices[i]
    return i, j, min_distances.max()


def _is_connected(adjacency: npt.NDArray[np.float32], i: int, j: int) -> bool:
    if adjacency[i, j]:
        return True

    connections = np.where(adjacency[i])[0]
    adjacency[i] = False
    adjacency[:, i] = False
    for idx in connections:
        if _is_connected(adjacency, idx, j):
            return True
    return False
