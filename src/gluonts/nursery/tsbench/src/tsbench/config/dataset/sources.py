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

import json
import shutil
import tempfile
import urllib.request as req
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast, Dict, List, Tuple
from zipfile import ZipFile
from gluonts.dataset.common import MetaData
from gluonts.dataset.repository._tsf_datasets import Dataset as MonashDataset
from gluonts.dataset.repository._tsf_datasets import (
    save_datasets,
    save_metadata,
)
from gluonts.dataset.repository._tsf_reader import TSFReader
from gluonts.dataset.repository.datasets import materialize_dataset
from tsbench.config.dataset.preprocessing.filters import (
    AbsoluteValueFilter,
    ConstantTargetFilter,
    EndOfSeriesCutFilter,
    MinLengthFilter,
)
from ._base import DatasetConfig
from .preprocessing import Filter, read_transform_write


@dataclass(frozen=True)
class GluonTsDatasetConfig(DatasetConfig):  # pylint: disable=abstract-method
    """
    A dataset configuration for datasets obtained directly via GluonTS.
    """

    def generate(self) -> None:
        if self.root.exists():
            return

        (self.root / "gluonts").mkdir(parents=True)

        # Download data and move to our own managed directory
        with tempfile.TemporaryDirectory() as directory:
            self._materialize(Path(directory))
            source = Path(directory) / self._gluonts_name

            # Copy and read metadata
            meta_file = self.root / "gluonts" / "metadata.json"
            shutil.copyfile(source / "metadata.json", meta_file)
            meta = MetaData.parse_file(meta_file)

            # Copy the data and apply filters
            filters = self._filters(
                self._prediction_length_multiplier
                * cast(int, meta.prediction_length)
            )
            read_transform_write(
                self.root / "gluonts" / "train" / "data.json",
                filters=filters
                + [EndOfSeriesCutFilter(cast(int, meta.prediction_length))],
                source=source / "train" / "data.json",
            )
            read_transform_write(
                self.root / "gluonts" / "val" / "data.json",
                filters=filters,
                source=source / "train" / "data.json",
            )

            # Although we increase the prediction length for the filters here, this does not
            # exclude any more data! The time series is only longer by the prediction length...
            read_transform_write(
                self.root / "gluonts" / "test" / "data.json",
                filters=self._filters(
                    (self._prediction_length_multiplier + 1)
                    * cast(int, meta.prediction_length)
                ),
                source=source / "test" / "data.json",
            )

    @property
    def _gluonts_name(self) -> str:
        return self.name()

    @property
    def _prediction_length_multiplier(self) -> int:
        # This is a legacy field and should NOT be overridden. It was accidentally set for the
        # preprocessing of some datasets though.
        return 0

    def _filters(self, prediction_length: int) -> List[Filter]:
        return [
            ConstantTargetFilter(prediction_length),
            AbsoluteValueFilter(1e18),
        ]

    def _materialize(self, directory: Path, regenerate: bool = False) -> None:
        materialize_dataset(
            self._gluonts_name, directory, regenerate=regenerate
        )


@dataclass(frozen=True)
class MonashDatasetConfig(GluonTsDatasetConfig):
    """
    A dataset configuration for datasets obtained through forecastingdata.org,
    the Monash Forecasting Repository.
    """

    @property
    def _prediction_length(self) -> int:
        raise NotImplementedError

    @property
    def _file(self) -> str:
        raise NotImplementedError

    @property
    def _record(self) -> str:
        raise NotImplementedError

    def _materialize(self, directory: Path, regenerate: bool = False) -> None:
        dataset = MonashDataset(self._file, self._record)
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            with ZipFile(dataset.download(temp_path)) as archive:
                archive.extractall(path=temp_path)

            # only one file is exptected
            reader = TSFReader(temp_path / archive.namelist()[0])
            meta, data = reader.read()

        path = directory / self._gluonts_name
        path.mkdir()

        # Save metadata and dataset (filling in missing timestamps)
        save_metadata(
            path,
            len(data),
            _get_frequency(meta.frequency),
            self._prediction_length,
        )

        data = [
            {**d, "start_timestamp": d.get("start_timestamp", "1970-01-01")}
            for d in data
        ]
        save_datasets(path, data, self._prediction_length)


@dataclass(frozen=True)
class M3DatasetConfig(GluonTsDatasetConfig):  # pylint: disable=abstract-method
    """
    A dataset configuration shared by all M3 datasets.
    """

    def generate(self) -> None:
        if self.root.exists():
            return

        # Download the .xls file
        target = Path.home() / ".mxnet" / "gluon-ts" / "datasets" / "M3C.xls"
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            opener = req.build_opener()
            opener.addheaders = [("User-Agent", "")]
            req.install_opener(opener)
            req.urlretrieve(
                "https://forecasters.org/data/m3comp/M3C.xls", target
            )

        # Run the generation
        super().generate()


@dataclass(frozen=True)
class KaggleDatasetConfig(GluonTsDatasetConfig):
    """
    A dataset configuration for datasets obtained directly from Kaggle.
    """

    def generate(self) -> None:
        # Check for the existence of the data
        if self.root.exists():
            return

        data_root = (
            Path.home() / ".mxnet" / "gluon-ts" / "datasets" / self.name()
        )
        if not data_root.exists():
            raise ValueError(
                f"download the dataset from Kaggle ({self._link}) and unzip it"
                f" into {data_root}"
            )

        # Extract the data and apply filters. The min length filter uses +3 as otherwise, catch22
        # features cannot be computed
        metadata, series = self._extract_data(data_root)
        filters = self._filters(metadata["prediction_length"]) + [
            MinLengthFilter(2 * metadata["prediction_length"] + 3)
        ]
        for f in filters:
            series = f(series)

        # Write everything to file
        test_dir = self.root / "gluonts" / "test"
        test_dir.mkdir(parents=True)

        with (self.root / "gluonts" / "metadata.json").open("w+") as f:
            json.dump(metadata, f)

        with (test_dir / "data.json").open("w+") as f:
            for item in series:
                json.dump(item, f)
                f.write("\n")

        # Create training and validation data
        read_transform_write(
            self.root / "gluonts" / "val" / "data.json",
            filters=[EndOfSeriesCutFilter(metadata["prediction_length"])],
            source=test_dir / "data.json",
        )
        read_transform_write(
            self.root / "gluonts" / "train" / "data.json",
            filters=[EndOfSeriesCutFilter(2 * metadata["prediction_length"])],
            source=test_dir / "data.json",
        )

    @property
    def _link(self) -> str:
        raise NotImplementedError

    def _extract_data(
        self, path: Path
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        raise NotImplementedError


# -------------------------------------------------------------------------------------------------


def _get_frequency(monash_frequency: str) -> str:
    parts = monash_frequency.split("_")
    assert len(parts) <= 2, "invalid frequency (too many underscores)"
    if len(parts) == 1:
        return _get_base(parts[0])
    return _get_multiple(parts[0]) + _get_base(parts[1])


def _get_base(part: str) -> str:
    if part.lower().startswith("m"):
        if part.lower() == "monthly":
            return "M"
        if part.lower() == "minutely":
            return "min"
        raise ValueError(f"invalid frequency base {part}")
    return part.upper()[0]


def _get_multiple(part: str) -> str:
    if part.isnumeric():
        return part
    if part == "half":
        return "0.5"
    raise ValueError(f"invalid multiple string {part}")
