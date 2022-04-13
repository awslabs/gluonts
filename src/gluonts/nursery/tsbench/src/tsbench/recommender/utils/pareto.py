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

from typing import List, Optional, Union
import numpy as np
import numpy.typing as npt
import scipy.stats as st


def pareto_efficiency_mask(X: npt.NDArray[np.float32]) -> np.ndarray:
    """
    Evaluates for each allocation in the provided array whether it is Pareto
    efficient. The costs are assumed to be improved by lowering them.

    Args:
        X: Array of shape [N, D] (N: number of allocations, D: number of costs) containing the
            allocations to check.

    Returns:
        A boolean array of shape [N], indicating for each allocation whether it is Pareto
            efficient.
    """
    # First, we assume that all allocations are Pareto efficient, i.e. not dominated
    mask = np.ones(X.shape[0], dtype=bool)
    # Then, we iterate over all allocations A and check which are dominated by then current
    # allocation A. If it is, we don't need to check it against another allocation.
    for i, allocation in enumerate(X):
        # Only consider allocation if it hasn't been dominated yet
        if mask[i]:
            # An allocation is dominated by A if all costs are equal or lower and at least one cost
            # is strictly lower. Using that definition, A cannot be dominated by itself.
            dominated = np.all(allocation <= X[mask], axis=1) * np.any(
                allocation < X[mask], axis=1
            )
            mask[mask] = ~dominated

    return mask


def epsilon_net_indices(
    X: npt.NDArray[np.float32], dim: Optional[int] = None
) -> np.ndarray:
    """
    Outputs an order of the items in the provided array such that the items are
    spaced well. This means that after choosing a seed item, the next item is
    chosen to be the farthest from the seed item. The third item is then chosen
    to maximize the distance to the existing points and so on.

    This algorithm is taken from "Nearest-Neighbor Searching and Metric Space Dimensions"
    (Clarkson, 2005, p.17).

    Args:
        X: Array of shape [N, D] (N: number of items, D: dimensionality) with the items to
            sparsify.
        dim: The index of the dimension which to use to choose the seed item. If `None`, an item is
            chosen at random, otherwise the item with the lowest value in the specified dimension
            is used.

    Returns:
        An array of shape [N] providing the item indices that define a sparsified order of the
            items.
    """
    indices = set(range(X.shape[0]))

    # Choose the seed item according to dim
    if dim is None:
        seed = np.random.choice(X.shape[0])
    else:
        seed = np.argmin(X, axis=0)[dim]

    # Initialize the order
    order = [seed]
    indices.remove(seed)

    # Iterate until all models have been chosen
    while indices:
        # Get the distance to all items that have already been chosen
        ordered_indices = list(indices)
        diff = (
            X[ordered_indices][:, None, :].repeat(len(order), axis=1)
            - X[order]
        )
        min_distances = np.linalg.norm(diff, axis=-1).min(-1)

        # Then, choose the one with the maximum distance to all points
        choice = ordered_indices[min_distances.argmax()]
        order.append(choice)
        indices.remove(choice)

    return np.array(order)


def argsort_nondominated(
    X: npt.NDArray[np.float32],
    dim: Optional[int] = None,
    max_items: Optional[int] = None,
    flatten: bool = True,
) -> Union[List[int], List[List[int]]]:
    """
    Performs a multi-objective sort by iteratively computing the Pareto front
    and sparsifying the items within the Pareto front. This is a non-dominated
    sort leveraging an epsilon-net. All metrics are normalized using quantile
    normalization.

    Args:
        X: Array of shape [N, D] (N: number of items, D: dimensionality) with the
            multi-dimensional items to sort.
        dim: The feature (metric) to prefer when ranking items within the Pareto front. If `None`,
            items are chosen randomly.
        max_items: The maximum number of items that should be returned. When this is `None`, all
            items are sorted.
        flatten: Whether to flatten the resulting array.

    Returns:
        The indices of the sorted items, either globally or within each of the Pareto fronts,
        depending on the value of `flatten`.
    """
    remaining = np.arange(X.shape[0])
    indices = []
    num_items = 0

    # First, we want to normalize the input by performing a quantile transform into the range
    # [0, 1]
    X = (st.rankdata(X, axis=0) - 1) / (X.shape[0] - 1)  # type: ignore

    # Iterate until max_items are reached or there are no items left
    while remaining.size > 0 and (max_items is None or num_items < max_items):
        # Compute the Pareto front and sort the items within
        pareto_mask = pareto_efficiency_mask(X[remaining])
        pareto_front = remaining[pareto_mask]
        pareto_order = epsilon_net_indices(X[pareto_front], dim=dim)

        # Add order to the indices
        indices.append(pareto_front[pareto_order].tolist())
        num_items += len(pareto_front)

        # Remove items in the Pareto front from the remaining items
        remaining = remaining[~pareto_mask]

    # Restrict the number of items returned and optionally flatten
    if max_items is not None:
        limit = max_items - sum(len(x) for x in indices[:-1])
        indices[-1] = indices[-1][:limit]
        if not indices[-1]:
            indices = indices[:-1]

    if flatten:
        return [i for ix in indices for i in ix]
    return indices
