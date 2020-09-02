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

from gluonts.dataset.artificial import constant_dataset
from gluonts.dataset.parallelized_loader import ShuffleIter
import itertools


# test if ShuffleIter would return a iterator of the same size of the base iterator
def test_shuffle_iter() -> None:
    # test with range
    data = [{str(i): str(i)} for i in range(20)]
    shuffled_data = ShuffleIter(
        base_iterator=iter(data), shuffle_buffer_length=10
    )
    assert len(list(shuffled_data)) == 20

    # test with constant gluonts dataset
    ds_info, train_ds, test_ds = constant_dataset()
    base_iter, base_iter_backup = itertools.tee(iter(train_ds), 2)
    shuffled_data = ShuffleIter(
        base_iterator=base_iter, shuffle_buffer_length=5
    )
    assert len(list(shuffled_data)) == len(list(base_iter_backup))
