import gzip
import json
import pickle
from pathlib import Path

from gluonts.dataset.common import Dataset, FileDataset
from gluonts.itertools import batcher


def pickle_dataset(ds: Dataset, *, freq: str, location: Path, batch_size=8):
    length = 0

    for batch_number, batch in enumerate(batcher(ds, batch_size), start=1):
        with gzip.open(location / f"{batch_number}.pckl.gz", "wb") as out_file:
            length += len(batch)
            pickle.dump(batch, out_file)

    with open(location / "metadata.json", "w") as metadata_file:
        json.dump(
            dict(
                freq=freq,
                length=length,
                num_batches=batch_number,
                batch_size=batch_size,
            ),
            metadata_file,
        )


def json_load_file(path):
    with path.open() as in_file:
        return json.load(in_file)


class PickledDataset(Dataset):
    def __init__(self, path: Path):
        self.path = path
        self.info = json_load_file(path / "metadata.json")

    def __iter__(self):
        for idx in range(1, self.info["num_batches"] + 1):
            with gzip.open(self.path / f"{idx}.pckl.gz", "rb") as in_file:
                yield from pickle.load(in_file)

    def __len__(self):
        return self.info["length"]
