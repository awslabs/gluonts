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


import functools
import glob
import random
from pathlib import Path
from typing import Iterator, List
from typing import NamedTuple

import rapidjson as json

from .common import Dataset, DataEntry, SourceContext
from .process import ProcessDataEntry


def load(file_obj):
    for line in file_obj:
        yield json.loads(line)


class Span(NamedTuple):
    path: Path
    line: int


class Line(NamedTuple):
    content: object
    span: Span


class JsonLinesFile:
    """
    An iterable type that draws from a JSON Lines file.

    Parameters
    ----------
    path
        Path of the file to load data from. This should be a valid
        JSON Lines file.
    """

    def __init__(self, path: Path, shuffle: bool = True) -> None:
        self.path = path
        self.shuffle = shuffle

    def __iter__(self):
        with open(self.path) as jsonl_file:
            lines = jsonl_file.read().splitlines()
            if self.shuffle:
                random.shuffle(lines)

            for line_number, raw in enumerate(lines, start=1):
                span = Span(path=self.path, line=line_number)
                try:
                    yield Line(json.loads(raw), span=span)
                except ValueError:
                    raise Exception(f"Could not read json line {line_number}, {raw}")

    def __len__(self):
        # 1MB
        BUF_SIZE = 1024 ** 2

        with open(self.path) as file_obj:
            read_chunk = functools.partial(file_obj.read, BUF_SIZE)
            return sum(chunk.count("\n") for chunk in iter(read_chunk, ""))


class FileDataset(Dataset):
    """
    Dataset that loads JSON Lines files contained in a path.

    Parameters
    ----------
    path
        Return list of path names that match path. Each file is considered
        and should be valid. A valid line in a file can be for
        instance: {"start": "2014-09-07", "target": [0.1, 0.2]}.
    freq
        Frequency of the observation in the time series.
        Must be a valid Pandas frequency.
    one_dim_target
        Whether to accept only univariate target time series.
    shuffle
        Whether to shuffle the time series when making the batches
    """

    def __init__(
        self, path: Path, freq: str, one_dim_target: bool = True, shuffle: bool = False
    ) -> None:
        self.shuffle = shuffle
        self.path = path
        self.process = ProcessDataEntry(freq, one_dim_target=one_dim_target)
        if not self.files():
            raise OSError(f"no valid file found via {path}")

    def __iter__(self) -> Iterator[DataEntry]:
        for path in self.files():
            for line in JsonLinesFile(path, self.shuffle):
                data = self.process(line.content)
                data["source"] = SourceContext(
                    source=line.span.path, row=line.span.line
                )
                yield data

    def __len__(self):
        return sum([len(JsonLinesFile(path)) for path in self.files()])

    def files(self) -> List[Path]:
        """
        List the files that compose the dataset.

        Returns
        -------
        List[Path]
            List of the paths of all files composing the dataset.
        """
        files = glob.glob(str(self.path))
        if self.shuffle:
            random.shuffle(files)
        return files

