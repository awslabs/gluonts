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


from __future__ import annotations
import warnings
import json
from typing import Dict, Union, Optional
from collections import OrderedDict

from .meters import Meter


class MeterDict(object):
    def __init__(
        self,
        meters: Optional[Dict[str, Meter]] = None,
        meterdicts: Optional[Dict[str, MeterDict]] = None,
    ) -> None:
        self._meters = OrderedDict()
        self._meterdicts = OrderedDict()
        if meters is not None:
            for name, meter in meters.items():
                self.register_meter(name, meter)
        if meterdicts is not None:
            for name, meterdict in meterdicts.items():
                self.register_meterdict(name, meterdict)

    def register_meter(self, name: str, meter: Meter) -> None:
        if "_meters" not in self.__dict__:
            raise AttributeError(
                "cannot assign meter before MeterDict.__init__() call"
            )
        elif "/" in name:
            raise KeyError("meter name cannot contain '/'")
        elif name == "":
            raise KeyError("meter name cannot be empty")
        elif hasattr(self, name) and name not in self._meters:
            raise KeyError(f"attribute {name} already exists")
        self._meters[name] = meter

    def register_meterdict(self, name: str, meterdict: MeterDict) -> None:
        if "_meterdicts" not in self.__dict__:
            raise AttributeError(
                "cannot assign meterdict before MeterDict.__init__() call"
            )
        elif "/" in name:
            raise KeyError("meterdict name cannot contain '/'")
        elif name == "":
            raise KeyError("meterdict name cannot be empty")
        elif hasattr(self, name) and name not in self._meters:
            raise KeyError(f"attribute {name} already exists")
        self._meterdicts[name] = meterdict

    def __setattr__(self, name: str, value):
        if isinstance(value, Meter):
            if name in self.__dict__:
                del self.__dict__[name]
            if name in self._meterdicts:
                del self._meterdicts[name]
            self.register_meter(name, value)
        elif isinstance(value, MeterDict):
            if name in self.__dict__:
                del self.__dict__[name]
            if name in self._meters:
                del self._meters[name]
            self.register_meterdict(name, value)
        else:
            super(MeterDict, self).__setattr__(name, value)

    def __getattr__(self, name: str):
        if "_meters" in self.__dict__:
            _meters = self.__dict__["_meters"]
            if name in _meters:
                return _meters[name]
        if "_meterdicts" in self.__dict__:
            _meterdicts = self.__dict__["_meterdicts"]
            if name in _meterdicts:
                return _meterdicts[name]
        raise AttributeError(f"{type(self).__name__} has no attribute {name}")

    def __delattr__(self, name: str):
        if name in self._meters:
            del self._meters[name]
        elif name in self._meterdicts:
            del self._meterdicts[name]
        else:
            object.__delattr__(name)

    def __setitem__(self, name: str, value):
        names = name.split("/", 1)
        if len(names) > 1:
            member = getattr(self, names[0])
            member[names[1]] = value
        else:
            setattr(self, names[0], value)

    def __getitem__(self, name: str):
        names = name.split("/", 1)
        item = getattr(self, names[0])
        if len(names) > 1:
            return item[names[1]]
        else:
            return item

    def __contains__(self, name: str):
        names = name.split("/", 1)
        if hasattr(self, names[0]):
            if len(names) > 1:
                return names[1] in getattr(self, names[0])
            else:
                return True
        else:
            return False

    def get(self, name: str, default=None):
        if name in self:
            return self[name]
        else:
            return default

    def _named_dicts(self, memo: Optional[set] = None, prefix: str = ""):
        if memo is None:
            memo = set()
        if self not in memo:
            memo.add(self)
            yield prefix, self
            if len(prefix) > 0:
                prefix += "/"
            for name, meterdict in self._meterdicts.items():
                for m in meterdict._named_dicts(memo, prefix + name):
                    yield m

    def _named_meters(self, prefix: str = ""):
        memo = set()
        for name_prefix, meterdict in self._named_dicts(prefix=prefix):
            for name, meter in meterdict._meters.items():
                if meter in memo:
                    continue
                memo.add(meter)
                name = (
                    name_prefix + ("/" if len(name_prefix) > 0 else "") + name
                )
                yield name, meter

    def restart(self):
        for _, meter in self._named_meters():
            meter.restart()

    @property
    def value(self) -> Dict:
        return {name: meter.value for name, meter in self._named_meters()}

    @property
    def best(self) -> Dict:
        return {name: meter.best for name, meter in self._named_meters()}

    def state_dict(self) -> Dict:
        return {
            name: meter.state_dict() for name, meter in self._named_meters()
        }

    def load_state_dict(self, state_dict: Dict) -> None:
        missing_keys = []
        for name, meter in self._named_meters():
            if name in state_dict:
                meter_state = state_dict.pop(name)
                meter.load_state_dict(meter_state)
            else:
                missing_keys.append(name)
        unexpected_keys = list(state_dict.keys())
        if len(missing_keys) > 0:
            warnings.warn(f"missing keys in state_dict: {missing_keys}")
        if len(unexpected_keys) > 0:
            warnings.warn(f"unexpected keys in state_dict: {unexpected_keys}")

    def __repr__(self) -> str:
        def _add_spaces(str_, n_spaces=4):
            return "\n".join(
                [(" " * n_spaces) + line for line in str_.split("\n")]
            )

        main_str = "\n".join(
            [f"{name}: {repr(meter)}" for name, meter in self._meters.items()]
        )
        child_str = "\n".join(
            [
                f"{name}:\n{_add_spaces(repr(meterdict))}"
                for name, meterdict in self._meterdicts.items()
            ]
        )
        if child_str:
            main_str = "\n".join([main_str, child_str])
        return main_str
