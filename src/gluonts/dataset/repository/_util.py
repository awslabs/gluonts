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

from typing import List, Union


def metadata(
    cardinality: Union[int, List[int]], freq: str, prediction_length: int
):
    if not isinstance(cardinality, list):
        cardinality = [cardinality]

    return {
        "freq": freq,
        "prediction_length": prediction_length,
        "feat_static_cat": [
            {"name": f"feat_static_cat_{i}", "cardinality": str(card)}
            for i, card in enumerate(cardinality)
        ],
    }


def request_retrieve_hook(tqdm):
    """Wraps tqdm instance, usable in request.urlretrieve
    the tqdm instance once you're done with it (easiest using `with` syntax).
    Example
    -------
    # >>> with tqdm(...) as _tqdm:
    # ...     request.urlretrieve(..., reporthook=request_retrieve_hook(_tqdm))
    """
    last_byte = 0

    def update_to(block=1, block_size=1, tsize=None):
        """
        block  : int, optional
            Number of blocks transferred so far [default: 1].
        block_size  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        nonlocal last_byte
        if tsize is not None:
            tqdm.total = tsize
        tqdm.update((block - last_byte) * block_size)
        last_byte = block

    return update_to
