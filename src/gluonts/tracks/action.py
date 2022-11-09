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

from typing import List

from dataclasses import dataclass


class Action:
    def apply(self, stream):
        raise NotImplementedError

    def apply_one(self, data):
        return list(self.apply([data]))

    def apply_schema(self, schema):
        raise NotImplementedError

    def requires(self):
        raise NotImplementedError

    def __add__(self, other):
        if isinstance(other, Pipeline):
            return Pipeline([self] + other.actions)

        return Pipeline([self, other])


@dataclass
class Pipeline(Action):
    actions: List[Action]

    def apply(self, stream):
        for action in self.actions:
            stream = action.apply(stream)

        return stream

    def apply_schema(self, schema):
        for action in self.actions:
            schema = action.apply(schema)

        return schema

    def __add__(self, other):
        if isinstance(other, Pipeline):
            return Pipeline(self.actions + other.actions)

        return Pipeline(self.actions + [other])


class Map(Action):
    def each(self, data):
        raise NotImplementedError

    def apply(self, stream):
        return map(self.each, stream)


class Identity(Map):
    def each(self, data):
        return data

    def apply_schema(self, schema):
        return schema


class Copy(Map):
    def each(self, data):
        return dict(data)

    def apply_schema(self, schema):
        return schema


class Filter(Action):
    def filter(self, data):
        raise NotImplementedError

    def apply_schema(self, schema):
        return schema

    def apply(self, data):
        return filter(self.filter, data)


@dataclass
class Remove(Map):
    name: str

    def each(self, data):
        del data[self.name]
        return data

    def apply_schema(self, schema):
        return schema.remove(self.name)
