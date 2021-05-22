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

from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Union,
    cast,
)

import numpy as np
import pandas as pd
import pydantic
from pandas.tseries.offsets import Tick

from gluonts.core.exception import GluonTSDataError
from gluonts.dataset import jsonl, util

# Dictionary used for data flowing through the transformations.
DataEntry = Dict[str, Any]
DataBatch = Dict[str, Any]

# TODO: change this maybe to typing_extensions.Protocol
# A Dataset is an iterable of DataEntry.
Dataset = Iterable[DataEntry]


class Timestamp(pd.Timestamp):
    # we need to sublcass, since pydantic otherwise converts the value into
    # datetime.datetime instead of using pd.Timestamp
    @classmethod
    def __get_validators__(cls):
        def conv(val):
            if isinstance(val, pd.Timestamp):
                return val
            else:
                return pd.Timestamp(val)

        yield conv


class BasicFeatureInfo(pydantic.BaseModel):
    name: str


class CategoricalFeatureInfo(pydantic.BaseModel):
    name: str
    cardinality: str


class MetaData(pydantic.BaseModel):
    freq: str = pydantic.Field(..., alias="time_granularity")  # type: ignore
    target: Optional[BasicFeatureInfo] = None

    feat_static_cat: List[CategoricalFeatureInfo] = []
    feat_static_real: List[BasicFeatureInfo] = []
    feat_dynamic_real: List[BasicFeatureInfo] = []
    feat_dynamic_cat: List[CategoricalFeatureInfo] = []

    prediction_length: Optional[int] = None

    class Config(pydantic.BaseConfig):
        allow_population_by_field_name = True


class SourceContext(NamedTuple):
    source: str
    row: int


class TrainDatasets(NamedTuple):
    """
    A dataset containing two subsets, one to be used for training purposes,
    and the other for testing purposes, as well as metadata.
    """

    metadata: MetaData
    train: Dataset
    test: Optional[Dataset] = None

    def save(self, path_str: str, overwrite=True) -> None:
        """
        Saves an TrainDatasets object to a JSON Lines file.

        Parameters
        ----------
        path_str
            Where to save the dataset.
        overwrite
            Whether to delete previous version in this folder.
        """
        import shutil
        from gluonts import json

        path = Path(path_str)

        if overwrite:
            shutil.rmtree(path, ignore_errors=True)

        def dump_line(f, line):
            f.write(json.dumps(line).encode("utf-8"))
            f.write("\n".encode("utf-8"))

        (path / "metadata").mkdir(parents=True)
        with open(path / "metadata/metadata.json", "wb") as f:
            dump_line(f, self.metadata.dict())

        (path / "train").mkdir(parents=True)
        with open(path / "train/data.json", "wb") as f:
            for entry in self.train:
                dump_line(f, serialize_data_entry(entry))

        if self.test is not None:
            (path / "test").mkdir(parents=True)
            with open(path / "test/data.json", "wb") as f:
                for entry in self.test:
                    dump_line(f, serialize_data_entry(entry))


class FileDataset(Dataset):
    """
    Dataset that loads JSON Lines files contained in a path.

    Parameters
    ----------
    path
        Path containing the dataset files. Each file is considered
        and should be valid to the exception of files starting with '.'
        or ending with '_SUCCESS'. A valid line in a file can be for
        instance: {"start": "2014-09-07", "target": [0.1, 0.2]}.
    freq
        Frequency of the observation in the time series.
        Must be a valid Pandas frequency.
    one_dim_target
        Whether to accept only univariate target time series.
    cache
        Indicates whether the dataset should be cached or not.
    """

    def __init__(
        self,
        path: Path,
        freq: str,
        one_dim_target: bool = True,
        cache: bool = False,
    ) -> None:
        self.cache = cache
        self.path = path
        self.process = ProcessDataEntry(freq, one_dim_target=one_dim_target)
        self._len_per_file = None

        if not self.files():
            raise OSError(f"no valid file found in {path}")

        # necessary, in order to preserve the cached datasets, in case caching was enabled
        self._json_line_files = [
            jsonl.JsonLinesFile(path=path, cache=cache)
            for path in self.files()
        ]

    def __iter__(self) -> Iterator[DataEntry]:
        for json_line_file in self._json_line_files:
            for line in json_line_file:
                data = self.process(line.content)
                data["source"] = SourceContext(
                    source=line.span.path, row=line.span.line
                )
                yield data

    # Returns array of the sizes for each subdataset per file
    def len_per_file(self):
        if self._len_per_file is None:
            len_per_file = [
                len(json_line_file) for json_line_file in self._json_line_files
            ]
            self._len_per_file = len_per_file
        return self._len_per_file

    def __len__(self):
        return sum(self.len_per_file())

    def files(self) -> List[Path]:
        """
        List the files that compose the dataset.

        Returns
        -------
        List[Path]
            List of the paths of all files composing the dataset.
        """
        return util.find_files(self.path, self.is_valid)

    @classmethod
    def is_valid(cls, path: Path) -> bool:
        # TODO: given that we only support json, should we also filter json
        # TODO: in the extension?
        return not (path.name.startswith(".") or path.name == "_SUCCESS")


class ListDataset(Dataset):
    """
    Dataset backed directly by a list of dictionaries.

    data_iter
        Iterable object yielding all items in the dataset.
        Each item should be a dictionary mapping strings to values.
        For instance: {"start": "2014-09-07", "target": [0.1, 0.2]}.
    freq
        Frequency of the observation in the time series.
        Must be a valid Pandas frequency.
    one_dim_target
        Whether to accept only univariate target time series.
    """

    def __init__(
        self,
        data_iter: Iterable[DataEntry],
        freq: str,
        one_dim_target: bool = True,
    ) -> None:
        self.process = ProcessDataEntry(freq, one_dim_target)
        self.list_data = list(data_iter)  # dataset always cached

    def __iter__(self) -> Iterator[DataEntry]:
        source_name = "list_data"
        # Basic idea is to split the dataset into roughly equally sized segments
        # with lower and upper bound, where each worker is assigned one segment
        bounds = util.get_bounds_for_mp_data_loading(len(self))
        for row_number, data in enumerate(self.list_data):
            if not bounds.lower <= row_number < bounds.upper:
                continue

            data = data.copy()
            data = self.process(data)
            data["source"] = SourceContext(source=source_name, row=row_number)
            yield data

    def __len__(self):
        return len(self.list_data)


class TimeZoneStrategy(Enum):
    ignore = "ignore"
    utc = "utc"
    error = "error"


# TODO: find out whether this is a duplicate
class ProcessStartField(pydantic.BaseModel):
    """
    Transform the start field into a Timestamp with the given frequency.

    Parameters
    ----------
    name
        Name of the field to transform.
    freq
        Frequency to use. This must be a valid Pandas frequency string.
    """

    class Config:
        arbitrary_types_allowed = True

    freq: Union[str, pd.DateOffset]
    name: str = "start"
    tz_strategy: TimeZoneStrategy = TimeZoneStrategy.error

    def __call__(self, data: DataEntry) -> DataEntry:
        try:
            timestamp = ProcessStartField.process(data[self.name], self.freq)
        except (TypeError, ValueError) as e:
            raise GluonTSDataError(
                f'Error "{e}" occurred, when reading field "{self.name}"'
            )

        if timestamp.tz is not None:
            if self.tz_strategy == TimeZoneStrategy.error:
                raise GluonTSDataError(
                    "Timezone information is not supported, "
                    f'but provided in the "{self.name}" field.'
                )
            elif self.tz_strategy == TimeZoneStrategy.utc:
                # align timestamp to utc timezone
                timestamp = timestamp.tz_convert("UTC")

            # removes timezone information
            timestamp = timestamp.tz_localize(None)

        data[self.name] = timestamp

        return data

    @staticmethod
    @lru_cache(maxsize=10000)
    def process(string: str, freq: str) -> pd.Timestamp:
        """Create timestamp and align it according to frequency."""

        timestamp = pd.Timestamp(string, freq=freq)

        # operate on time information (days, hours, minute, second)
        if isinstance(timestamp.freq, Tick):
            return pd.Timestamp(
                timestamp.floor(timestamp.freq), timestamp.freq
            )

        # since we are only interested in the data piece, we normalize the
        # time information
        timestamp = timestamp.replace(
            hour=0, minute=0, second=0, microsecond=0, nanosecond=0
        )

        return timestamp.freq.rollforward(timestamp)


class ProcessTimeSeriesField:
    """
    Converts a time series field identified by `name` from a list of numbers
    into a numpy array.

    Constructor parameters modify the conversion logic in the following way:

    If `is_required=True`, throws a `GluonTSDataError` if the field is not
    present in the `Data` dictionary.

    If `is_cat=True`, the array type is `np.int32`, otherwise it is
    `np.float32`.

    If `is_static=True`, asserts that the resulting array is 1D,
    otherwise asserts that the resulting array is 2D. 2D dynamic arrays of
    shape (T) are automatically expanded to shape (1,T).

    Parameters
    ----------
    name
        Name of the field to process.
    is_required
        Whether the field must be present.
    is_cat
        Whether the field refers to categorical (i.e. integer) values.
    is_static
        Whether the field is supposed to have a time dimension.
    """

    # TODO: find a fast way to assert absence of nans.

    def __init__(
        self, name, is_required: bool, is_static: bool, is_cat: bool
    ) -> None:
        self.name = name
        self.is_required = is_required
        self.req_ndim = 1 if is_static else 2
        self.dtype = np.int32 if is_cat else np.float32

    def __call__(self, data: DataEntry) -> DataEntry:
        value = data.get(self.name, None)
        if value is not None:
            value = np.asarray(value, dtype=self.dtype)

            if self.req_ndim != value.ndim:
                raise GluonTSDataError(
                    f"Array '{self.name}' has bad shape - expected "
                    f"{self.req_ndim} dimensions, got {value.ndim}."
                )

            data[self.name] = value

            return data
        elif not self.is_required:
            return data
        else:
            raise GluonTSDataError(
                f"Object is missing a required field `{self.name}`"
            )


class ProcessDataEntry:
    def __init__(self, freq: str, one_dim_target: bool = True) -> None:
        # TODO: create a FormatDescriptor object that can be derived from a
        # TODO: Metadata and pass it instead of freq.
        # TODO: In addition to passing freq, the descriptor should be carry
        # TODO: information about required features.
        self.trans = cast(
            List[Callable[[DataEntry], DataEntry]],
            [
                ProcessStartField(freq=freq),
                # The next line abuses is_static=True in case of 1D targets.
                ProcessTimeSeriesField(
                    "target",
                    is_required=True,
                    is_cat=False,
                    is_static=one_dim_target,
                ),
                ProcessTimeSeriesField(
                    "feat_dynamic_cat",
                    is_required=False,
                    is_cat=True,
                    is_static=False,
                ),
                ProcessTimeSeriesField(
                    "dynamic_feat",  # backwards compatible
                    is_required=False,
                    is_cat=False,
                    is_static=False,
                ),
                ProcessTimeSeriesField(
                    "feat_dynamic_real",
                    is_required=False,
                    is_cat=False,
                    is_static=False,
                ),
                ProcessTimeSeriesField(
                    "past_feat_dynamic_real",
                    is_required=False,
                    is_cat=False,
                    is_static=False,
                ),
                ProcessTimeSeriesField(
                    "feat_static_cat",
                    is_required=False,
                    is_cat=True,
                    is_static=True,
                ),
                ProcessTimeSeriesField(
                    "feat_static_real",
                    is_required=False,
                    is_cat=False,
                    is_static=True,
                ),
            ],
        )

    def __call__(self, data: DataEntry) -> DataEntry:
        for t in self.trans:
            data = t(data)
        return data


def load_datasets(
    metadata: Path, train: Path, test: Optional[Path]
) -> TrainDatasets:
    """
    Loads a dataset given metadata, train and test path.

    Parameters
    ----------
    metadata
        Path to the metadata file
    train
        Path to the training dataset files.
    test
        Path to the test dataset files.

    Returns
    -------
    TrainDatasets
        An object collecting metadata, training data, test data.
    """
    meta = MetaData.parse_file(Path(metadata) / "metadata.json")
    train_ds = FileDataset(path=train, freq=meta.freq)
    test_ds = FileDataset(path=test, freq=meta.freq) if test else None

    return TrainDatasets(metadata=meta, train=train_ds, test=test_ds)


def serialize_data_entry(data):
    """
    Encode the numpy values in the a DataEntry dictionary into lists so the
    dictionary can be JSON serialized.

    Parameters
    ----------
    data
        The dictionary to be transformed.

    Returns
    -------
    Dict
        The transformed dictionary, where all fields where transformed into
        strings.
    """

    def serialize_field(field):
        if isinstance(field, np.ndarray):
            # circumvent https://github.com/micropython/micropython/issues/3511
            nan_ix = np.isnan(field)
            field = field.astype(np.object_)
            field[nan_ix] = "NaN"
            return field.tolist()
        if isinstance(field, (int, float)):
            return field
        return str(field)

    return {k: serialize_field(v) for k, v in data.items() if v is not None}
