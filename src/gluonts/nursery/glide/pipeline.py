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

from toolz.functoolz import compose_left

from gluonts.nursery.glide import Apply, ParApply
from gluonts.nursery.glide.util import lift


def lifted(fns):
    return tuple(map(lift, fns))


class Pipeline:
    def __init__(self, fns=()):
        self.fns = tuple(fns)

    def apply(self, parts, *args, **kwargs):
        return Apply(compose_left(*self.fns), parts, *args, **kwargs)

    def parapply(self, parts, *args, **kwargs):
        return ParApply(compose_left(*self.fns), parts, *args, **kwargs)

    def and_then(self, *fns):
        return Pipeline(self.fns + fns)

    def and_then_each(self, *fns):
        return self.and_then(*lifted(fns))

    def but_first(self, *fns):
        return Pipeline(fn, fns + self.fns)

    def but_first_each(self, *fns):
        return self.but_first(*lifted(fns))

    def __add__(self, other):
        return Pipeline(self.fns + other.fns)
