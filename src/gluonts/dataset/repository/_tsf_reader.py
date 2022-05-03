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

from datetime import datetime
from distutils.util import strtobool
from multiprocessing import cpu_count
from types import SimpleNamespace
from typing import Dict

import numpy as np
from toolz import compose_left

from gluonts import json
from gluonts.exceptions import GluonTSDataError
from gluonts.nursery import glide

parse_bool = compose_left(strtobool, bool)


def parse_attribute(ty, value: str):
    if ty == "numeric":
        return int(value)

    if ty == "string":
        return value

    if ty == "date":
        return datetime.strptime(value, "%Y-%m-%d %H-%M-%S")

    raise AttributeError(ty)


def frequency_converter(freq: str):
    parts = freq.split("_")
    if len(parts) == 1:
        return convert_base(parts[0])
    if len(parts) == 2:
        return convert_multiple(parts[0]) + convert_base(parts[1])
    raise ValueError(f"Invalid frequency string {freq}.")


BASE_FREQ_TO_PANDAS_OFFSET: Dict[str, str] = {
    "seconds": "S",
    "minutely": "T",
    "minutes": "T",
    "hourly": "H",
    "hours": "H",
    "daily": "D",
    "days": "D",
    "weekly": "W",
    "weeks": "W",
    "monthly": "M",
    "months": "M",
    "quarterly": "Q",
    "quarters": "Q",
    "yearly": "Y",
    "years": "Y",
}


def convert_base(text: str) -> str:
    try:
        return BASE_FREQ_TO_PANDAS_OFFSET[text]
    except KeyError:
        raise GluonTSDataError(
            f'"{text}" is not recognized as a frequency string'
        )


def convert_multiple(text: str) -> str:
    if text.isnumeric():
        return text
    if text == "half":
        return "0.5"
    raise ValueError(f"Unknown frequency multiple {text}.")


class TSFReader:
    def __init__(
        self,
        path,
        target_name="target",
    ):
        self.path = path
        self.target_name = target_name

        self.meta = SimpleNamespace(columns={})

    def read(self):
        with open(self.path, encoding="latin1") as in_file:
            # strip whitespace
            lines = map(str.strip, in_file)

            # ignore all lines starting with #
            lines = filter(lambda line: not line.startswith("#"), lines)

            data_tag_found = self._read_header(lines)
            assert data_tag_found, "Missing @data tag."
            assert self.meta.columns, (
                "Missing attribute section. Attribute section must come before"
                " data."
            )

            assert self.target_name not in self.meta.columns
            self.meta.columns[self.target_name] = None

            data = self._read_data_section(lines)

            return self.meta, data

    def _read_header(self, lines):
        for line in lines:
            assert line.startswith("@")
            stop = self._tag(line[1:])

            if stop:
                return True

        return False

    def _read_data_section(self, lines):
        # Enumerate here to keep the indices
        data = list(enumerate(lines))
        result = glide.imap_unordered(
            self._read_data, data, num_workers=cpu_count(), batch_size=8092
        )
        # Sort by index here to ensure that the order is deterministic
        return [x[1] for x in sorted(result, key=lambda x: x[0])]

    def _read_data(self, data):
        idx, line = data
        parts = line.split(":")

        assert len(parts) == len(
            self.meta.columns
        ), "Missing attributes/values in series."

        *attributes, target = parts

        record = {}

        record[self.target_name] = self._data_target(target)

        for (column, ty), attr in zip(self.meta.columns.items(), attributes):
            record[column] = parse_attribute(ty, attr)

        return idx, record

    def _data_target(self, s):
        s = s.replace("?", '"nan"')

        values = json.loads(f"[{s}]")
        assert values, (
            "A given series should contains a set of comma separated numeric"
            " values. At least one numeric value should be there in a series."
            " Missing values should be indicated with ? symbol"
        )

        return np.array(values, dtype=float)

    def _tag(self, line):
        fn_by_tag = {
            "attribute": self._tag_attribute,
            "frequency": self._tag_frequency,
            "horizon": self._tag_horizon,
            "missing": self._tag_missing,
            "equallength": self._tag_equallength,
            "data": self._tag_data,
        }
        tag, *args = line.split(" ")

        if tag not in fn_by_tag:
            return

        return fn_by_tag[tag](*args)

    def _tag_attribute(self, name, ty):
        self.meta.columns[name] = ty

    def _tag_frequency(self, frequency):
        self.meta.frequency = frequency

    def _tag_horizon(self, horizon):
        self.meta.forecast_horizon = horizon

    def _tag_missing(self, missing):
        self.meta.has_missing_values = parse_bool(missing)

    def _tag_equallength(self, equallength):
        self.meta.has_equal_length = parse_bool(equallength)

    def _tag_data(self):
        return True
