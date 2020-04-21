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

# Standard library imports
import itertools
import shutil
from collections import defaultdict
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Union,
    Generic,
    TypeVar,
    Type,
)

# Third-party imports
import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
import pydantic
import ujson as json
from pandas.tseries.offsets import Tick

# First-party imports
from gluonts.core.exception import GluonTSDataError
from gluonts.dataset import jsonl, util

# Dictionary used for data flowing through the transformations.
from gluonts.dataset.jsonl import open_file

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


class TimeSeriesItem(pydantic.BaseModel):
    class Config:
        arbitrary_types_allowed = True
        json_encoders = {np.ndarray: np.ndarray.tolist}

    start: Timestamp
    target: np.ndarray
    item: Optional[str] = None

    feat_static_cat: List[int] = []
    feat_static_real: List[float] = []
    feat_dynamic_cat: List[List[int]] = []
    feat_dynamic_real: List[List[float]] = []

    # A dataset can use this field to include information about the origin of
    # the item (e.g. the file name and line). If an exception in a
    # transformation occurs the content of the field will be included in the
    # error message (if the field is set).
    metadata: dict = {}

    @pydantic.validator("target", pre=True)
    def validate_target(cls, v):
        return np.asarray(v)

    def __eq__(self, other: Any) -> bool:
        # we have to overwrite this function, since we can't just compare
        # numpy ndarrays, but have to call all on it
        if isinstance(other, TimeSeriesItem):
            return (
                self.start == other.start
                and (self.target == other.target).all()
                and self.item == other.item
                and self.feat_static_cat == other.feat_static_cat
                and self.feat_static_real == other.feat_static_real
                and self.feat_dynamic_cat == other.feat_dynamic_cat
                and self.feat_dynamic_real == other.feat_dynamic_real
            )
        return False

    def gluontsify(self, metadata: "MetaData") -> dict:
        data: dict = {
            "item": self.item,
            "start": self.start,
            "target": self.target,
        }

        if metadata.feat_static_cat:
            data["feat_static_cat"] = self.feat_static_cat
        if metadata.feat_static_real:
            data["feat_static_real"] = self.feat_static_real
        if metadata.feat_dynamic_cat:
            data["feat_dynamic_cat"] = self.feat_dynamic_cat
        if metadata.feat_dynamic_real:
            data["feat_dynamic_real"] = self.feat_dynamic_real

        return data


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


class Channel(pydantic.BaseModel):
    metadata: Path
    train: Path
    test: Optional[Path] = None

    def get_datasets(self) -> "TrainDatasets":
        return load_datasets(self.metadata, self.train, self.test)


class TrainDatasets(NamedTuple):
    """
    A dataset containing two subsets, one to be used for training purposes,
    and the other for testing purposes, as well as metadata.
    """

    metadata: MetaData
    train: Dataset
    test: Optional[Dataset] = None


T = TypeVar("T")


class FieldType(Generic[T]):
    is_optional: bool

    def __call__(self, v: Any) -> T:
        raise NotImplementedError()

    def __eq__(self, other):
        raise NotImplementedError()

    def is_compatible(self, v: Any) -> bool:
        raise NotImplementedError()


class NumpyArrayField(FieldType[np.ndarray]):
    def __init__(
        self,
        is_optional: bool,
        dtype: Type = np.float32,
        ndim: Optional[int] = None,
    ) -> None:
        self.dtype = dtype
        self.ndim = ndim
        self.is_optional = is_optional

    def __eq__(self, other):
        if not isinstance(other, NumpyArrayField):
            return False
        return (
            self.dtype == other.dtype
            and self.ndim == other.ndim
            and self.is_optional == other.is_optional
        )

    def __repr__(self):
        return f"NumpyArrayField(dtype={self.dtype!r}, ndim={self.ndim!r}, is_optional={self.is_optional!r})"

    def __call__(self, value: Any) -> np.ndarray:
        value = np.asarray(value, dtype=object)
        value = value.astype(self.dtype)
        if self.ndim is not None and self.ndim != value.ndim:
            raise GluonTSDataError(
                f"expected array with dimension {self.ndim}, but got {value.ndim}."
            )
        return value

    def is_compatible(self, value: Any) -> bool:
        if not isinstance(value, (list, tuple, np.ndarray)):
            return False

        try:
            x = np.asarray(value)
        except:
            return False

        # int types
        if self.dtype in [np.int32, np.int64]:
            if x.dtype.kind != "i":
                return False
            return self.ndim is None or x.ndim == self.ndim

        # float types
        try:
            x = np.asarray(value, dtype=self.dtype)
        except:
            return False
        return self.ndim is None or x.ndim == self.ndim


class TimeZoneStrategy(Enum):
    ignore = "ignore"
    utc = "utc"
    error = "error"


Freq = Union[str, pd.DateOffset]


class PandasTimestampField(FieldType[pd.Timestamp]):
    def __init__(
        self,
        is_optional: bool,
        freq: Freq,
        tz_strategy: TimeZoneStrategy = TimeZoneStrategy.error,
    ) -> None:
        self.freq = to_offset(freq) if isinstance(freq, str) else freq
        self.tz_strategy = tz_strategy
        self.is_optional = is_optional

    def __eq__(self, other):
        if not isinstance(other, PandasTimestampField):
            return False
        return (
            self.freq == other.freq
            and self.tz_strategy == other.tz_strategy
            and self.is_optional == other.is_optional
        )

    def __repr__(self):
        return f"PandasTimestampField(freq={self.freq!r}, tz_strategy={self.tz_strategy!r}, is_optional={self.is_optional!r})"

    def __call__(self, value: Any) -> pd.Timestamp:
        timestamp = PandasTimestampField._process(value, self.freq)

        if timestamp.tz is not None:
            if self.tz_strategy == TimeZoneStrategy.error:
                raise GluonTSDataError("Timezone information is not supported")
            elif self.tz_strategy == TimeZoneStrategy.utc:
                # align timestamp to utc timezone
                timestamp = timestamp.tz_convert("UTC")

            # removes timezone information
            timestamp = timestamp.tz_localize(None)
        return timestamp

    @staticmethod
    @lru_cache(maxsize=10000)
    def _process(string: str, freq: pd.DateOffset) -> pd.Timestamp:
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

    def is_compatible(self, v: Any) -> bool:
        if not isinstance(v, (str, pd.Timestamp)):
            return False
        try:
            self(v)
        except:
            return False
        return True


class AnyField(FieldType[Any]):
    def __init__(self, is_optional: bool):
        self.is_optional = is_optional

    def __eq__(self, other):
        if not isinstance(other, AnyField):
            return False
        return self.is_optional == other.is_optional

    def __repr__(self):
        return f"AnyField(is_optional={self.is_optional!r})"

    def __call__(self, value: Any) -> Any:
        return value

    def is_compatible(self, v: Any) -> bool:
        return True


class Schema:
    def __init__(self, fields: Dict[str, FieldType]) -> None:
        self.fields = fields
        # select fields that should be handled in the loop, because
        # - they are non optional
        # - or they are not AnyFields
        self._fields_for_processing = {
            k: f
            for k, f in self.fields.items()
            if not f.is_optional or not isinstance(f, AnyField)
        }

    def __eq__(self, other):
        if self.fields.keys() != other.fields.keys():
            return False
        return all(self.fields[k] == other.fields[k] for k in self.fields)

    def __repr__(self):
        return (
            "Schema(fields={"
            + ", ".join(f"'{k}':{v}" for k, v in self.fields.items())
            + "})"
        )

    def __call__(self, d: Dict[str, Any]) -> None:
        """
        Applies the schema to a data dict. The dictionary is updated in place.
        """
        for field_name, field_type in self._fields_for_processing.items():
            try:
                value = d[field_name]
            except KeyError:
                if not field_type.is_optional:
                    raise GluonTSDataError(
                        f"field {field_name} is not optional but key does not occur in the data"
                    )
                else:
                    continue
            try:
                d[field_name] = field_type(value)
            except Exception as e:
                raise GluonTSDataError(
                    f"Error when processing field {field_name} using {field_type}"
                ) from e

    @staticmethod
    def infer(
        entries: List[Dict[str, Any]], freq: Optional[Freq] = None
    ) -> "Schema":
        """
        Infers the schema from the passed entries
        """
        observed_keys: Dict[str, int] = defaultdict(int)
        guessed_freq = to_offset(freq) if freq is not None else None
        for entry in entries:
            if freq is None:
                if not "freq" in entry:
                    raise GluonTSDataError(
                        "freq not provided and could not be found in data entry"
                    )
            elif "freq" in entry:
                found_freq = to_offset(entry["freq"])
                if guessed_freq is not None and found_freq != guessed_freq:
                    raise GluonTSDataError(
                        f"freq differs for different entries: {guessed_freq} != {found_freq}"
                    )
                else:
                    guessed_freq = found_freq
            for key in entry:
                observed_keys[key] += 1
        is_optional = {
            key: n < len(entries) for key, n in observed_keys.items()
        }

        freq = freq if freq is not None else guessed_freq

        fields = {}
        for key in observed_keys:
            candidate_types = [
                PandasTimestampField(freq=freq, is_optional=is_optional[key]),
                NumpyArrayField(dtype=np.int32, is_optional=is_optional[key]),
                NumpyArrayField(
                    dtype=np.float32, is_optional=is_optional[key]
                ),
                AnyField(is_optional=is_optional[key]),
            ]

            for entry in entries:
                if not key in entry:
                    continue
                value = entry[key]
                candidate_types = [
                    ct for ct in candidate_types if ct.is_compatible(value)
                ]

            fields[key] = candidate_types[0]
        return Schema(fields=fields)


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
    schema
        An optional schema that will be checked if provided.
        Otherwise the schema is inferred from the data.
    cache
        Indicates whether the dataset should be cached or not.
    """

    def __init__(
        self,
        path: Path,
        freq: Optional[str] = None,
        schema: Optional[Schema] = None,
        cache: bool = False,
    ) -> None:
        self.cache = cache
        self.path = path
        self.freq = freq
        self._len = None

        if not self.files():
            raise OSError(f"no valid file found in {path}")
        if schema is not None:
            self.schema = schema
        else:
            line_gen = (
                line
                for file in self.files()
                for line in jsonl.JsonLinesFile(path=file)
            )
            entries = [e.content for e in itertools.islice(line_gen, 10)]
            self.schema = Schema.infer(entries, freq=self.freq)

        # necessary, in order to preserve the cached datasets, in case caching was enabled
        self._json_line_files = [
            jsonl.JsonLinesFile(path=path, cache=cache)
            for path in self.files()
        ]

    def __iter__(self) -> Iterator[DataEntry]:
        for json_line_file in self._json_line_files:
            for line in json_line_file:
                data = line.content
                self.schema(data)
                data["source"] = SourceContext(
                    source=line.span.path, row=line.span.line
                )
                yield data

    def __len__(self):
        if self._len is None:
            len_sum = sum(
                [len(jsonl.JsonLinesFile(path=path)) for path in self.files()]
            )
            self._len = len_sum
        return self._len

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
        return not (path.name.startswith(".") or path.name == "_SUCCESS")


class ListDataset(Dataset):
    """
    Dataset backed directly by an list of dictionaries.

    data_iter
        Iterable object yielding all items in the dataset.
        Each item should be a dictionary mapping strings to values.
        For instance: {"start": "2014-09-07", "target": [0.1, 0.2]}.
    freq
        Frequency of the observation in the time series.
        Must be a valid Pandas frequency.
    schema
        An optional schema that will be checked if provided.
        Otherwise the schema is inferred from the data.
    """

    def __init__(
        self,
        data_iter: Iterable[DataEntry],
        freq: Optional[Freq],
        schema: Optional[Schema] = None,
    ) -> None:
        self.list_data = list(data_iter)
        if schema is not None:
            self.schema = schema
        else:
            self.schema = Schema.infer(self.list_data[:10], freq=freq)

    def __iter__(self) -> Iterator[DataEntry]:
        source_name = "list_data"
        # Basic idea is to split the dataset into roughly equally sized segments
        # with lower and upper bound, where each worker is assigned one segment
        segment_size = int(len(self) / util.MPWorkerInfo.num_workers)

        for row_number, data in enumerate(self.list_data):
            lower_bound = util.MPWorkerInfo.worker_id * segment_size
            upper_bound = (
                (util.MPWorkerInfo.worker_id + 1) * segment_size
                if util.MPWorkerInfo.worker_id + 1
                != util.MPWorkerInfo.num_workers
                else len(self)
            )
            if not lower_bound <= row_number < upper_bound:
                continue
            self.schema(data)
            data["source"] = SourceContext(source=source_name, row=row_number)
            yield data

    def __len__(self):
        return len(self.list_data)


def get_standard_schema(freq: str, one_dim_target: bool = True) -> Schema:
    return Schema(
        dict(
            start=PandasTimestampField(freq=freq, is_optional=False),
            target=NumpyArrayField(
                dtype=np.float32,
                ndim=1 if one_dim_target else 2,
                is_optional=False,
            ),
            feat_dynamic_cat=NumpyArrayField(
                dtype=np.int32, ndim=2, is_optional=True
            ),
            feat_dynamic_real=NumpyArrayField(
                dtype=np.float32, ndim=2, is_optional=True
            ),
            feat_static_cat=NumpyArrayField(
                dtype=np.int32, ndim=1, is_optional=True
            ),
            feat_static_real=NumpyArrayField(
                dtype=np.float32, ndim=1, is_optional=True
            ),
        )
    )


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


def save_datasets(
    dataset: TrainDatasets, path_str: str, gzip: bool = False, overwrite=True
) -> None:
    """
    Saves an TrainDatasets object to a JSON Lines file.

    Parameters
    ----------
    dataset
        The training datasets.
    path_str
        Where to save the dataset.
    overwrite
        Whether to delete previous version in this folder.
    """
    path = Path(path_str)

    if overwrite:
        shutil.rmtree(path, ignore_errors=True)

    def dump_line(f, line):
        f.write(json.dumps(line).encode("utf-8"))
        f.write("\n".encode("utf-8"))

    (path / "metadata").mkdir(parents=True)
    with open(path / "metadata/metadata.json", "wb") as f:
        dump_line(f, dataset.metadata.dict())

    gz_ending = ".gz" if gzip else ""

    (path / "train").mkdir(parents=True)
    with open_file(path / f"train/data.json{gz_ending}", "wb") as f:
        for entry in dataset.train:
            dump_line(f, serialize_data_entry(entry))

    if dataset.test is not None:
        (path / "test").mkdir(parents=True)
        with open_file(path / f"test/data.json{gz_ending}", "wb") as f:
            for entry in dataset.test:
                dump_line(f, serialize_data_entry(entry))


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
        return str(field)

    return {k: serialize_field(v) for k, v in data.items() if v is not None}
