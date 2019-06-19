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

# Standard library imports
import shutil
import tempfile
from contextlib import contextmanager
from typing import Optional, Tuple

import numpy as np


@contextmanager
def TemporaryDirectory():
    name = tempfile.mkdtemp()
    try:
        yield name
    finally:
        shutil.rmtree(name)


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i : i + n]


def empirical_cdf(
    samples: np.ndarray, num_bins: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the empricial cdf from the given samples.

    Parameters
    ----------
    samples
        Tensor of samples of shape (num_samples, batch_shape)

    Returns
    -------
    Tensor
        Emprically calculated cdf values. shape (num_bins, batch_shape)

    Tensor
        Bin edges corresponding to the cdf values. shape (num_bins + 1, batch_shape)
    """

    # calculate histogram separately for each dimension in the batch size
    cdfs = []
    edges = []

    batch_shape = samples.shape[1:]
    agg_batch_dim = np.prod(batch_shape)

    samples = samples.reshape((samples.shape[0], -1))

    for i in range(agg_batch_dim):
        s = samples[:, i]
        bins = np.linspace(s.min(), s.max(), num_bins + 1)
        hist, edge = np.histogram(s, bins=bins)
        cdfs.append(np.cumsum(hist / len(s)))
        edges.append(edge)

    empirical_cdf = np.stack(cdfs, axis=-1).reshape(num_bins, *batch_shape)
    edges = np.stack(edges, axis=-1).reshape(num_bins + 1, *batch_shape)
    return empirical_cdf, edges
