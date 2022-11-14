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

from dataclasses import dataclass
from typing import Any, List

from gluonts import itertools as it


class Action:
    def apply(self, stream):
        raise NotImplementedError

    def apply_one(self, data):
        return list(self.apply([data]))

    def apply_schema(self, schema):
        raise NotImplementedError

    def bind(self, schema):
        return Bind(self, schema, self.apply_schema(schema))

    def requires(self):
        raise NotImplementedError

    def __add__(self, other):
        return other.__radd__(self)

    def __radd__(self, other):
        if isinstance(other, Pipeline):
            return Pipeline(other.actions + [self])

        return Pipeline([other, self])


@dataclass
class Bind(Action):
    action: Action
    input_schema: "Schema"
    output_schema: "Schema"

    def apply(self, stream):
        return self.action.apply(stream)


@dataclass
class Pipeline(Action):
    actions: List[Action]

    def apply(self, stream):
        for action in self.actions:
            stream = action.apply(stream)

        return stream

    def apply_schema(self, schema):
        for action in self.actions:
            schema = action.apply_schema(schema)

        return schema

    def __radd__(self, other):
        if isinstance(other, Pipeline):
            return Pipeline(other.actions + self.actions)

        return Pipeline([other] + self.actions)


class Map(Action):
    def each(self, data):
        raise NotImplementedError

    def apply(self, stream):
        return it.Map(self.each, stream)


class Identity(Map):
    def __add__(self, other):
        return other

    def __radd__(self, other):
        return other

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
        return it.Filter(self.filter, data)


@dataclass
class Set(Map):
    name: str
    value: Any

    def each(self, data):
        data[self.name] = self.value
        return data

    def apply_schema(self, schema):
        return schema.add(self.name, type(self.value))


@dataclass
class Remove(Map):
    name: str

    def each(self, data):
        del data[self.name]
        return data

    def apply_schema(self, schema):
        return schema.remove(self.name)
