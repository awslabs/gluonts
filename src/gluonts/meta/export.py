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
            path = ".".join([self.name, path])

            parts = path.rsplit(":", 1)
            if len(parts) == 1:
                module_name = parts[0]
                name = ""
            else:
                module_name, name = parts

            module = import_module(module_name)

            if name:
                value = getattr(module, name)
                if hasattr(value, "__module__"):
                    value.__module__ = self.name
            else:
                value = module

            setattr(self.module, name, value)

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


def re_export(module, *names, **kwargs):
    """Import sub-paths and assign `__module__` on definitions.

    Often there is a difference between the intended external interface and
    internal code structure. For example, the ``Predictor`` class is defined
    in ``gluonts/model/predictor.py``, but the canonical path is
    ``gluonts.model.Predictor`` and not ``gluonts.model.predictor.Predictor``.

    The desired behaviour can be achieved using the following in
    ``model/__init__.py``::

        __all__ = re_export(
            __name__,
            "estimator:Estimator",
            "predictor:Predictor",
            ...
        )

    """

    names = list(names)

    for base, paths in kwargs.items():
        for path in paths:
            names.append(":".join([base, path]))

    exports = Exports(module)
    for name in names:
        exports.re_export(name)
    return exports
