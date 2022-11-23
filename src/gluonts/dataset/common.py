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
import logging
import shutil
from pathlib import Path
from types import ModuleType
from typing import Callable, List, NamedTuple, Optional, Union, cast

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset

import pydantic

from gluonts import json
from gluonts.itertools import Cached, Map
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.schema import Translator
from gluonts.exceptions import GluonTSDataError


from . import Dataset, DatasetCollection, DataEntry, DataBatch  # noqa
from . import jsonl, DatasetWriter


arrow: Optional[ModuleType]

try:
    from . import arrow
except ImportError:
    arrow = None


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
    A dataset containing two subsets, one to be used for training purposes, and
    the other for testing purposes, as well as metadata.
    """

    metadata: MetaData
    train: Dataset
    test: Optional[Dataset] = None

    def save(
        self,
        path_str: str,
        writer: DatasetWriter,
        overwrite=False,
    ) -> None:
        """
        Saves an TrainDatasets object to a JSON Lines file.

        Parameters
        ----------
        path_str
            Where to save the dataset.
        overwrite
            Whether to delete previous version in this folder.
        """
        path = Path(path_str)

        if overwrite:
            shutil.rmtree(path, ignore_errors=True)

        path.mkdir(parents=True)
        with open(path / "metadata.json", "wb") as out_file:
            json.bdump(self.metadata.dict(), out_file, nl=True)

        train = path / "train"
        train.mkdir(parents=True)
        writer.write_to_folder(self.train, train)

        if self.test is not None:
            test = path / "test"
            test.mkdir(parents=True)
            writer.write_to_folder(self.test, test)


def infer_file_type(path):
    suffix = "".join(path.suffixes)

    if suffix in jsonl.JsonLinesFile.SUFFIXES:
        return jsonl.JsonLinesFile(path)

    if arrow is not None and suffix in arrow.File.SUFFIXES:
        return arrow.File.infer(path)

    return None


def _glob(path: Path, pattern="*", levels=1):
    if levels is not None:
        levels -= 1

    for subpath in path.glob(pattern):
        if subpath.is_dir():
            if levels != 0:
                yield from _glob(subpath, pattern, levels)
        else:
            yield subpath


def FileDataset(
    path: Path,
    freq: str,
    one_dim_target: bool = True,
    cache: bool = False,
    use_timestamp: bool = False,
    loader_class=None,
    ignore=False,
    pattern="*",
    levels=1,
    translate=None,
) -> Dataset:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    if path.is_dir():
        subpaths = _glob(path, pattern, levels)

        datasets = [
            FileDataset(
                subpath,
                freq,
                one_dim_target,
                cache,
                use_timestamp,
                loader_class=loader_class,
                ignore=True,
            )
            for subpath in subpaths
        ]

        return DatasetCollection(
            [dataset for dataset in datasets if dataset is not None]
        )

    assert path.is_file()

    if loader_class is None:
        loader = infer_file_type(path)
        if loader is None:
            message = f"Cannot infer loader for {path}."
            if ignore:
                logging.warning(message)
                return None  # type: ignore

            raise ValueError(message)
    else:
        loader = loader_class(path)

    return _FileDataset(
        loader, freq, one_dim_target, cache, use_timestamp, translate
    )


def _FileDataset(
    dataset: Dataset,
    freq: str,
    one_dim_target: bool = True,
    cache: bool = False,
    use_timestamp: bool = False,
    translate: Optional[dict] = None,
) -> Dataset:
    process = ProcessDataEntry(
        freq, one_dim_target=one_dim_target, use_timestamp=use_timestamp
    )

    if translate is not None:
        dataset = cast(Dataset, Map(Translator.parse(translate), dataset))

    dataset = cast(Dataset, Map(process, dataset))

    if cache:
        dataset = cast(Dataset, Cached(dataset))

    return dataset


def ListDataset(
    data_iter: Dataset,
    freq: str,
    one_dim_target: bool = True,
    use_timestamp: bool = False,
    translate: Optional[dict] = None,
) -> List[DataEntry]:
    """
    Dataset backed directly by a list of dictionaries.

    Parameters
    ----------
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

    if translate is not None:
        data_iter = Map(Translator.parse(translate), data_iter)

    return list(
        Map(
            ProcessDataEntry(to_offset(freq), one_dim_target, use_timestamp),
            data_iter,
        )
    )


@functools.lru_cache(10_000)
def _as_period(val, freq):
    return pd.Period(val, freq)


# TODO: find out whether this is a duplicate
class ProcessStartField(pydantic.BaseModel):
    """
    Transform the start field into a Period with the given frequency.

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
    use_timestamp: bool = False
    name: str = FieldName.START

    def __call__(self, data: DataEntry) -> DataEntry:
        try:
            if self.use_timestamp:
                data[self.name] = pd.Timestamp(data[self.name])
            else:
                data[self.name] = _as_period(data[self.name], self.freq)
        except (TypeError, ValueError) as e:
            raise GluonTSDataError(
                f'Error "{e}" occurred, when reading field "{self.name}"'
            ) from e

        return data


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
    def __init__(
        self,
        freq: str,
        one_dim_target: bool = True,
        use_timestamp: bool = False,
    ) -> None:
        # TODO: create a FormatDescriptor object that can be derived from a
        # TODO: Metadata and pass it instead of freq.
        # TODO: In addition to passing freq, the descriptor should be carry
        # TODO: information about required features.
        self.trans = cast(
            List[Callable[[DataEntry], DataEntry]],
            [
                ProcessStartField(freq=freq, use_timestamp=use_timestamp),
                # The next line abuses is_static=True in case of 1D targets.
                ProcessTimeSeriesField(
                    FieldName.TARGET,
                    is_required=True,
                    is_cat=False,
                    is_static=one_dim_target,
                ),
                ProcessTimeSeriesField(
                    FieldName.FEAT_DYNAMIC_CAT,
                    is_required=False,
                    is_cat=True,
                    is_static=False,
                ),
                ProcessTimeSeriesField(
                    FieldName.FEAT_DYNAMIC_REAL_LEGACY,  # backwards compatible
                    is_required=False,
                    is_cat=False,
                    is_static=False,
                ),
                ProcessTimeSeriesField(
                    FieldName.FEAT_DYNAMIC_REAL,
                    is_required=False,
                    is_cat=False,
                    is_static=False,
                ),
                ProcessTimeSeriesField(
                    FieldName.PAST_FEAT_DYNAMIC_REAL,
                    is_required=False,
                    is_cat=False,
                    is_static=False,
                ),
                ProcessTimeSeriesField(
                    FieldName.FEAT_STATIC_CAT,
                    is_required=False,
                    is_cat=True,
                    is_static=True,
                ),
                ProcessTimeSeriesField(
                    FieldName.FEAT_STATIC_REAL,
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
    metadata: Path,
    train: Path,
    test: Optional[Path],
    one_dim_target: bool = True,
    cache: bool = False,
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
    one_dim_target
        Whether to load FileDatasets as univariate target time series.
    cache
        Indicates whether the FileDatasets should be cached or not.

    Returns
    -------
    TrainDatasets
        An object collecting metadata, training data, test data.
    """
    meta = MetaData.parse_file(Path(metadata) / "metadata.json")
    train_ds = FileDataset(
        path=train, freq=meta.freq, one_dim_target=one_dim_target, cache=cache
    )
    test_ds = (
        FileDataset(
            path=test,
            freq=meta.freq,
            one_dim_target=one_dim_target,
            cache=cache,
        )
        if test
        else None
    )

    return TrainDatasets(metadata=meta, train=train_ds, test=test_ds)
