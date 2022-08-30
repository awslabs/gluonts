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

import typing

from mypy.plugin import Plugin, ClassDefContext
from mypy.plugins.dataclasses import dataclass_class_maker_callback


class GluonTSPlugin(Plugin):
    def get_class_decorator_hook(
        self, fullname: str
    ) -> typing.Optional[typing.Callable[[ClassDefContext], None]]:

        if fullname == "gluonts.core.serde._dataclass.dataclass":
            return dataclass_class_maker_callback
        return None


def plugin(version: str) -> typing.Type[Plugin]:
    return GluonTSPlugin
