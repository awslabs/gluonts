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


from importlib import import_module
from types import ModuleType

from dataclasses import dataclass, field


@dataclass
class Exports:
    name: str
    module: ModuleType = field(init=False)
    exports: dict = field(default_factory=dict)

    def __post_init__(self):
        self.module = import_module(self.name)

    def re_export(self, *paths):
        for path in paths:
            if path.startswith("."):
                path = self.name + path

            module_name, *name = path.rsplit(".", 1)

            if name:
                name = name[0]
            else:
                name = ""

            module = import_module(module_name)
            value = getattr(module, name)

            setattr(self.module, name, value)

            if hasattr(value, "__module__"):
                value.__module__ = self.name

            self.exports[name] = value

    def definition(self, cls):
        self.exports[cls.__name__] = cls
        return cls

    def register(self, name, value):
        self.exports[name] = value
        return value

    def __iter__(self):
        yield from self.exports

    def __len__(self):
        return len(self.exports)

    def __getitem__(self, index):
        return list(self.exports)[index]


def re_export(module, *names):
    exports = Exports(module)
    for name in names:
        exports.re_export(name)
    return exports
