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

from typing import Dict
import numpy as np
import pandas as pd
import scipy.stats as st


def compute_ranks(candidates: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Computes the ranks of the provided candidates for each column in the
    provided dataframe. If candidates show the same values, ranks are tied.

    Args:
        candidates: A mapping from candidate names to performances. The indexes of all candidates
            must align and ranks are computed for each index value.

    Returns:
        A data frame containing the results of all candidates and datasets. The data frame has a
            multi-index with the candidate name on the first level and the dataset on the second.
    """
    _check_candidates(candidates)
    ref = next(iter(candidates.values()))

    # Sort data frames such that we can use NumPy operations to compute ranks
    sorted_candidates = [
        {"name": k, "df": d[ref.columns].sort_index()}
        for k, d in candidates.items()
    ]
    arr = np.stack([c["df"].to_numpy() for c in sorted_candidates], axis=0)  # type: ignore
    ranks = st.rankdata(arr, axis=0, method="min")

    # Extract data frames from ranks again
    sorted_index = ref.index.sort_values()

    index = pd.MultiIndex.from_tuples(
        [
            t
            for candidate in sorted_candidates
            for t in [(candidate["name"], item) for item in sorted_index]
        ],
        names=["candidate", "example"],
    )
    return pd.DataFrame(
        np.concatenate(ranks), index=index, columns=ref.columns
    )


def _check_candidates(candidates: Dict[str, pd.DataFrame]) -> None:
    ref = next(iter(candidates.values()))
    assert all(
        set(c.index) == set(ref.index) for c in candidates.values()
    ), "Candidates have differing indices."
    assert all(
        set(c.columns) == set(ref.columns) for c in candidates.values()
    ), "Candidates have differing columns."
