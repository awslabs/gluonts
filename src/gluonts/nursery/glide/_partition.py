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

from collections.abc import Sequence
from functools import singledispatch


def divide_into(length, n):
    base_size, r = divmod(length, n)

    partition_length = [base_size] * n

    for idx in range(r):
        partition_length[idx] += 1

    return partition_length


@singledispatch
def partition(xs, n):
    raise NotImplementedError(
        f"Type {type(xs)} has not implemeted `partition`."
    )


@partition.register
def partition_sequence(xs: Sequence, n):
    slices = divide_into(len(xs), n)

    start = 0
    partitions = []
    for slice_length in slices:
        partitions.append(xs[start : start + slice_length])
        start += slice_length

    return partitions
