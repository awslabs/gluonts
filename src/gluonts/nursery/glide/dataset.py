import functools
import ujson as json
from pathlib import Path

from toolz.itertoolz import take
import numpy as np

from gluonts.nursery import glide


class JsonLinesFile:
    def __init__(self, path: Path, _seek=None, _num_lines=None) -> None:
        self.path = path
        self._seek = _seek
        self._num_lines = _num_lines
        self._len = None

    def _get_line_starts(self):
        with self.path.open() as file_obj:
            n = 0
            for line in file_obj:
                yield n
                n += len(line)

    def __iter__(self):
        with open(self.path) as jsonl_file:
            if self._seek:
                jsonl_file.seek(self._seek)

            for raw in take(self._num_lines, jsonl_file):
                yield json.loads(raw)

    def __len__(self):
        if self._num_lines:
            return self._num_lines

        if self._len is None:
            # 1MB
            BUF_SIZE = 1024 ** 2

            with open(self.path, "rb") as file_obj:
                read_chunk = functools.partial(file_obj.read, BUF_SIZE)
                file_len = sum(
                    chunk.count(b"\n") for chunk in iter(read_chunk, b"")
                )
                self._len = file_len
        return self._len


@glide.partition.register
def partition(xs: JsonLinesFile, n):
    line_starts = list(xs._get_line_starts())
    partition_sizes = glide.divide_into(len(line_starts), n)

    current_line = 0
    files = []
    for idx, num_lines in enumerate(partition_sizes):
        files.append(
            JsonLinesFile(
                xs.path, _seek=line_starts[idx], _num_lines=num_lines
            )
        )
        current_line += num_lines

    return files
