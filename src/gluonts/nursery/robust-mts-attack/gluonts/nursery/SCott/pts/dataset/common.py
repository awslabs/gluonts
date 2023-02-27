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


from typing import Any, Dict, Iterable, NamedTuple, List, Optional

import pandas as pd
from pydantic import BaseModel

# Dictionary used for data flowing through the transformations.
DataEntry = Dict[str, Any]

# A Dataset is an iterable of DataEntry.
Dataset = Iterable[DataEntry]


class SourceContext(NamedTuple):
    source: str
    row: int


class FieldName:
    """
    A bundle of default field names to be used by clients when instantiating
    transformer instances.
    """

    ITEM_ID = "item_id"

    START = "start"
    TARGET = "target"

    FEAT_STATIC_CAT = "feat_static_cat"
    FEAT_STATIC_REAL = "feat_static_real"
    FEAT_DYNAMIC_CAT = "feat_dynamic_cat"
    FEAT_DYNAMIC_REAL = "feat_dynamic_real"

    FEAT_TIME = "time_feat"
    FEAT_CONST = "feat_dynamic_const"
    FEAT_AGE = "feat_dynamic_age"

    OBSERVED_VALUES = "observed_values"
    IS_PAD = "is_pad"
    FORECAST_START = "forecast_start"


class CategoricalFeatureInfo(BaseModel):
    name: str
    cardinality: str


class BasicFeatureInfo(BaseModel):
    name: str


class MetaData(BaseModel):
    freq: str = None
    target: Optional[BasicFeatureInfo] = None

    feat_static_cat: List[CategoricalFeatureInfo] = []
    feat_static_real: List[BasicFeatureInfo] = []
    feat_dynamic_real: List[BasicFeatureInfo] = []
    feat_dynamic_cat: List[CategoricalFeatureInfo] = []

    prediction_length: Optional[int] = None


class TrainDatasets(NamedTuple):
    """
    A dataset containing two subsets, one to be used for training purposes,
    and the other for testing purposes, as well as metadata.
    """

    metadata: MetaData
    train: Dataset
    test: Optional[Dataset] = None


class DateConstants:
    """
    Default constants for specific dates.
    """

    OLDEST_SUPPORTED_TIMESTAMP = pd.Timestamp(1800, 1, 1, 12)
    LATEST_SUPPORTED_TIMESTAMP = pd.Timestamp(2200, 1, 1, 12)
