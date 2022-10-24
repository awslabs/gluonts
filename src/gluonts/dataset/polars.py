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

from dataclasses import dataclass, field, InitVar
from functools import partial
from operator import methodcaller
from typing import Union, List, Optional

# import pandas as pd
import polars as pl

from gluonts.itertools import Map
from .pandas import _column_as_start
from .schema import Translator


@dataclass
class LongDataset:
    df: pl.DataFrame
    item_id: Union[str, List[str]]
    timestamp: Optional[str] = None
    freq: Optional[str] = None

    assume_sorted: bool = False
    translate: InitVar[Optional[dict]] = None
    translator: Optional[Translator] = field(default=None, init=False)

    use_partition: bool = False
    unchecked: bool = False

    def __post_init__(self, translate):
        if (self.timestamp is None) != (self.freq is None):
            raise ValueError(
                "Either both `timestamp` and `freq` have to be "
                "provided or neither."
            )

        if translate is not None:
            self.translator = Translator.parse(translate, drop=True)
        else:
            self.translator = None

    def _pop_item_id(self, dct):
        if isinstance(self.item_id, list):
            dct["item_id"] = ", ".join(
                dct.pop(column)[0] for column in self.item_id
            )
        else:
            dct["item_id"] = dct.pop(self.item_id)[0]

        return dct

    def __iter__(self):
        if self.use_partition:
            dataset = self.df.partition_by(self.item_id)
        else:
            dataset = self.df.groupby(self.item_id)

        if not self.assume_sorted:
            sort_by = [self.item_id]
            if self.timestamp is not None:
                sort_by.append(self.timestamp)
                dataset = Map(methodcaller("sort", by=sort_by), dataset)

        dataset = Map(methodcaller("to_dict", as_series=True), dataset)
        dataset = Map(self._pop_item_id, dataset)

        if self.translator is not None:
            dataset = Map(self.translator, dataset)

        if self.timestamp is not None:
            dataset = Map(
                partial(
                    _column_as_start,
                    column=self.timestamp,
                    freq=self.freq,
                    unchecked=self.unchecked,
                ),
                dataset,
            )

        yield from dataset

        # we were successful to iterate once over the dataset
        # so no more need to check more
        self.unchecked = True

    def __len__(self):
        return len(self.df.groupby(self.item_id).count())
