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

from toolz.curried import do


class Bind:
    def __init__(self, fn, stream):
        self.fn = fn
        self.stream = stream

    def __iter__(self):
        yield from self.fn.apply(self.stream)


class Pipeline:
    def __call__(self, stream):
        return Bind(self, stream)

    def apply(self, stream):
        # this will be overwritten in the sub-classes
        return stream

    def flat_map(self, fn):
        return self + FlatMap(fn)

    def map(self, fn):
        return self + Map(fn)

    def filter(self, fn):
        return self + Filter(fn)

    def and_then(self, other):
        return Chain(self, other)

    def __add__(self, other):
        return self.and_then(other)


class Chain(Pipeline):
    def __init__(self, *transformations):
        self.transformations = transformations

    def apply(self, stream):
        stream = stream
        for fn in self.transformations:
            stream = fn(stream)

        return stream


class FlatMap(Pipeline):
    def __init__(self, fn):
        self.fn = fn

    def apply(self, stream):
        for el in stream:
            yield from self.fn(el)


class Filter(FlatMap):
    def __init__(self, fn):
        self.fn = fn

    def apply(self, stream):
        for el in stream:
            if self.fn(el):
                yield el


class Map(Pipeline):
    def __init__(self, fn):
        self.fn = fn

    def apply(self, stream):
        yield from map(self.fn, stream)


class Repeat(Pipeline):
    def __init__(self, amount=None):
        self.amount = amount

    def apply(self, stream):
        while self.amount is None:
            yield from stream
        else:
            for _ in range(self.amount):
                yield from stream


class Cache(Pipeline):
    def __init__(self):
        self.cache = None

    def apply(self, stream):
        if self.cache is None:
            self.cache = []
            for element in stream:
                yield element
                self.cache.append(element)
        else:
            yield from self.cache
