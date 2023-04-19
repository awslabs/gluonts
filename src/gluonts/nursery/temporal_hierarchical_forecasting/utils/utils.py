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


from typing import Iterator, List

import numpy as np
import pandas as pd

from gluonts.core.component import validated
from gluonts.model.forecast import SampleForecast
from gluonts.mx import Tensor


class TemporalHierarchy:
    @validated()
    def __init__(self, agg_multiples: List[int], freq_strs: List[str]):
        self.num_leaves = agg_multiples[0]
        self.agg_multiples = agg_multiples
        self.freq_strs = freq_strs
        self.agg_mat = TemporalHierarchy._get_agg_mat(
            agg_multiples=agg_multiples
        )

    @property
    def _adj_mat(self):
        """
        Adjacency matrix for the temporal hierarchy.

        This is created in the following way:
            * First get indices of source vertices and destination vertices:
            source vertices are the aggregated nodes and destination vertices are their children in that order;
            e.g., for 15-min hierarchy, `source_ix` = [0, 1, 2] and `dest_ix` = [[1, 2], [3, 4], [5, 6]],
            i.e., 0 --> [1, 2], 1 --> [3, 4], 2 --> [5, 6].
            * Once we have these two lists, we can simply do `adj[s, t] = 1`, for `s` in `source_ix` and `t` `dest_ix`.
            * This is made symmetric by adding `adj[t, s] = 1`.

        :return:
        """
        num_nodes_running_sum = np.cumsum(self.num_nodes_per_level)

        # for 15-min hierarchy: [[1, 2], [3, 4, 5, 6]] (we ignore the root).
        node_ix_level_wise = [
            list(range(i, j))
            for i, j in zip(
                num_nodes_running_sum[:-1], num_nodes_running_sum[1:]
            )
        ]

        dest_ix = []
        for n, nodes_ix in zip(
            self.num_nodes_per_level[:-1], node_ix_level_wise
        ):
            # split `node_ix` list into `n` equal parts;
            # this relies on the definition of temporal hierarchy: equal splits during disaggregation!
            # for 15-min hierarchy: [1, 2] --> [1, 2]; [3, 4, 5, 6] --> [3, 4], [5, 6]
            dest_ix.extend(np.array_split(nodes_ix, n))

        total_num_nodes = np.sum(self.num_nodes_per_level)
        source_ix = list(range(total_num_nodes - self.num_leaves))

        adj = np.zeros((total_num_nodes, total_num_nodes))
        for s, t in zip(source_ix, dest_ix):
            adj[s, t] = 1
            adj[t, s] = 1

        return adj

    @property
    def _hierarchical_adj_mat(self):
        mat_accumulate = np.triu(self._adj_mat)
        mat_distribute = np.tril(self._adj_mat)
        col_sums = mat_distribute.sum(axis=0)
        mat_distribute[:, col_sums > 0] = mat_distribute[
            :, col_sums > 0
        ] / mat_distribute[:, col_sums > 0].sum(axis=0, keepdims=True)
        I = np.eye(mat_accumulate.shape[0])

        return (mat_accumulate + mat_distribute + I) / 3.0

    def adj_mat(self, option: str = "hierarchical"):
        assert option in [
            "standard_plus_identity",  # A + I
            "normalized_with_identity",  # D^{-1} (A + I), where D is the diagonal matrix of (A + I)
            "symmetric_normalized_with_identity",  # D^{-1/2} (A + I) D^{-1/2}, D is the diagonal matrix of (A + I)
            "standard",  # A
            "normalized",  # D^{-1} A, where D is the diagonal matrix of A
            "symmetric_normalized",  # D^{-1/2} A D^{-1/2}, D is the diagonal matrix of A
            "hierarchical",
        ]
        if option == "standard":
            return self._adj_mat
        if option == "standard_plus_identity":
            A = self._adj_mat
            return A + np.eye(A.shape[0])
        if option == "normalized":
            A = self._adj_mat
            D = np.diag(A.sum(axis=1))
            D_inv = np.linalg.inv(D)
            A_normalized = np.matmul(D_inv, A)
            return A_normalized
        if option == "normalized_with_identity":
            A = self._adj_mat
            A_hat = A + np.eye(A.shape[0])
            D = np.diag(A_hat.sum(axis=1))
            D_inv = np.linalg.inv(D)
            A_normalized_with_identity = np.matmul(D_inv, A_hat)
            return A_normalized_with_identity
        if option == "symmetric_normalized":
            A = self._adj_mat
            sqrt_D = np.diag(np.sqrt(A.sum(axis=1)))
            D_inv_sqrt = np.linalg.inv(sqrt_D)
            A_symmetric_normalized = np.matmul(
                D_inv_sqrt, np.matmul(A, D_inv_sqrt)
            )
            return A_symmetric_normalized
        if option == "symmetric_normalized_with_identity":
            A = self._adj_mat
            A_hat = A + np.eye(A.shape[0])
            sqrt_D = np.diag(np.sqrt(A_hat.sum(axis=1)))
            D_inv_sqrt = np.linalg.inv(sqrt_D)
            A_symmetric_normalized_with_identity = np.matmul(
                D_inv_sqrt, np.matmul(A_hat, D_inv_sqrt)
            )
            return A_symmetric_normalized_with_identity
        if option == "hierarchical":
            return self._hierarchical_adj_mat

    @property
    def num_nodes_per_level(self):
        return [
            self.num_leaves // agg_multiple
            for agg_multiple in self.agg_multiples
        ]

    @property
    def nodes(self):
        """
        Returns node structure representing the hierarchy as defined in the hts package.
        Examples:
            ["1D", "8H", "1H"] --> [3, [8, 8, 8]]
            ["1H", "30min", "1min"] --> [2, [30, 30]]

        Note that the nodes structure does not work if any node is *not* formed by grouping its immediate descendants,
        e.g., ["6H", "4H", "2H"].

        To know the exact structure of nodes see the help:
        Hierarchical: https://stackoverflow.com/questions/13579292/how-to-use-hts-with-multi-level-hierarchies
        Grouped: https://robjhyndman.com/hyndsight/gts/

        :return:
        """
        agg_multiples = np.array(self.agg_multiples)
        num_children = agg_multiples[:-1] // agg_multiples[1:]
        num_nodes_at_agg_levels = self.num_nodes_per_level[:-1]

        nodes = [
            [nc] * num_nodes if num_nodes > 1 else nc
            for nc, num_nodes in zip(num_children, num_nodes_at_agg_levels)
        ]
        return nodes

    @staticmethod
    def _get_agg_mat(agg_multiples: List[int]):
        num_leaves = agg_multiples[0]

        rows = []
        for agg_multiple in agg_multiples:
            num_rows = num_leaves // agg_multiple
            for i in range(num_rows):
                row = np.zeros(num_leaves)
                row[i * agg_multiple : (i + 1) * agg_multiple] = 1
                rows.append(row)

        return np.vstack(rows)


def _check_freqs(deltas: List[pd.Timedelta]):
    """
    Checks if all lower frequencies can be aggregated by highest frequency.
    """
    assert all(
        [d % deltas[-1] == pd.Timedelta("0H") for d in deltas]
    ), "Lower frequencies are not multiples of highest frequency."


def freqs_to_agg_mulitples(freq_strs: List[str]) -> List[int]:
    """
    Returns aggregation multiples that are used to construct
    aggregation matrix.

    Parameters
    ----------
    freq_strs : List[str]
        ordered list of frequencies. e.g ["1D", "8H", "1H"]
    """
    deltas = [pd.Timedelta(freq) for freq in freq_strs]
    _check_freqs(deltas)
    return [d // deltas[-1] for d in deltas]


def to_TemporalHierarchy(freq_strs: List[str]) -> TemporalHierarchy:
    agg_multiples = freqs_to_agg_mulitples(freq_strs)
    return TemporalHierarchy(agg_multiples=agg_multiples, freq_strs=freq_strs)


def agg_series(seq: Tensor, agg_multiple: int):
    batch_size, seq_length = seq.shape[0:2]

    return seq.reshape(
        batch_size,
        seq_length // agg_multiple,
        agg_multiple,
        -1,
    ).sum(axis=2)


def unpack_forecasts(
    forecast_at_all_levels_it: Iterator[SampleForecast],
    temporal_hierarchy: TemporalHierarchy,
    target_temporal_hierarchy: TemporalHierarchy,
    num_samples: int = 100,
) -> Iterator[SampleForecast]:
    """

    :param forecast_at_all_levels_it:
        Each element of the iterator has `SampleForecast` object with samples of shape:
            (`num_samples`, `prediction_length_at_root`, `total_num_nodes`),
        where the order of the last axis is level-wise left-right ordering of the hierarchy.
    :param temporal_hierarchy:
    :param num_samples:
        Used only for asserting shapes.
    :return:
    This will get the forecasts in this order:
        (ts_1_6M, ts_1_2M, ts_1_1M, ts_2_6M, ts_2_2M, ts_2_1M, ...)

    """

    num_nodes_per_level = temporal_hierarchy.num_nodes_per_level

    for forecast in forecast_at_all_levels_it:
        # We iterate over each time series and unpack forecasts for that time series at different
        # time granularites at the same time.
        samples_at_all_levels = forecast.samples
        assert num_samples == forecast.num_samples

        start_ix = 0
        for num_nodes, agg_freq, agg_multiple in zip(
            num_nodes_per_level,
            temporal_hierarchy.freq_strs,
            temporal_hierarchy.agg_multiples,
        ):
            end_ix = start_ix + num_nodes

            forecast_start_date = pd.Period(
                str(forecast.start_date),
                freq=(agg_multiple * forecast.start_date.freq).freqstr,
            )
            agg_forecast = SampleForecast(
                samples=samples_at_all_levels[..., start_ix:end_ix].reshape(
                    num_samples, -1
                ),
                start_date=forecast_start_date,
                item_id=forecast.item_id,
                info=forecast.info,
            )
            start_ix = end_ix

            if agg_multiple in target_temporal_hierarchy.agg_multiples:
                yield agg_forecast

        # Make sure we used all the samples!
        assert start_ix == samples_at_all_levels.shape[-1]


def get_ts_at_all_levels(
    ts_it: Iterator[pd.Series],
    temporal_hierarchy: TemporalHierarchy,
    prediction_length: int,
    target_temporal_hierarchy: TemporalHierarchy,
):
    for ts in ts_it:
        for agg_freq, agg_multiple in zip(
            temporal_hierarchy.freq_strs, temporal_hierarchy.agg_multiples
        ):
            if agg_multiple in target_temporal_hierarchy.agg_multiples:
                # We need to align the forecast start date for all aggregation levels.
                # This is possible if the time series length is multiple of `prediction_length`.
                ts = ts[-(len(ts) // prediction_length) * prediction_length :]

                agg_ts_values = ts.values.reshape(
                    len(ts) // agg_multiple, agg_multiple
                ).sum(axis=1)

                # Start date is same for the aggregated frequency:
                # For example if you aggregate 5-minutes data, that starts at 00:10, to hourly then the index of hourly
                # data will be 00:10, 01:10, 02:10, ...
                agg_ts_start_date = ts.index[0]
                agg_ts_index = pd.period_range(
                    start=agg_ts_start_date,
                    periods=len(agg_ts_values),
                    freq=(agg_multiple * ts.index[0].freq).freqstr,
                )
                agg_ts = pd.Series(agg_ts_values, index=agg_ts_index)

                yield agg_ts


def naive_reconcilation_mat(S: np.ndarray, nodes: List):
    """
    Returns the (average) reconciliation matrix that reconciles forecasts from all levels.
    In particular, it first computes mapping matrices `G` to reconcile base forecasts of any chosen level.
    The G matrix is the one described in the book: https://otexts.com/fpp2/mapping-matrices.html
    After obtining such `G` matrices for each level (G_1, G_2, ..., G_k), it computes the final `projection` or the
    `reconciliation` matrix as:
                P = S (G_1 + G_2 + ... + G_k) / k.
    This way it would use the un-reconciled forecasts of all nodes in the hierarchy to arrive at the final coherent
    forecasts.
    Parameters
    ----------
    S
        Summation or aggregation matrix. Shape: (total_num_time_series, num_base_time_series)
    nodes
        Node structure representing the hierarchy as defined in the hts package.
        To know the exact structure of nodes see the help:
        Hierarchical: https://stackoverflow.com/questions/13579292/how-to-use-hts-with-multi-level-hierarchies
        Grouped: https://robjhyndman.com/hyndsight/gts/
    Returns
    -------
    Numpy array of shape: (total_num_time_series, total_num_time_series)
    """
    num_ts, num_bottom_series = S.shape
    num_nodes_per_level = [1] + [np.asscalar(np.sum(n)) for n in nodes]
    cum_num_nodes_per_level = np.cumsum(num_nodes_per_level)

    def mapping_matrix_at_level(level: int):
        start_ix = 0 if level == 0 else cum_num_nodes_per_level[level - 1]
        end_ix = cum_num_nodes_per_level[level]

        M = np.zeros((num_bottom_series, num_ts))
        M[:, start_ix:end_ix] = S[start_ix:end_ix, :].transpose()

        # equal proportions
        row_sum = M[:, start_ix:end_ix].sum(axis=0)
        M[:, start_ix:end_ix] = M[:, start_ix:end_ix] / row_sum[None, :]
        return M

    mapping_matrices = np.array(
        [
            mapping_matrix_at_level(level=level)
            for level in range(len(cum_num_nodes_per_level))
        ]
    )

    mean_mapping_matrix = np.mean(mapping_matrices, axis=0)
    reconciliation_mat = np.matmul(S, mean_mapping_matrix)

    return reconciliation_mat
