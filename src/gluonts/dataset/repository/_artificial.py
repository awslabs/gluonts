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

from pathlib import Path
from typing import Optional

from gluonts.dataset import DatasetWriter
from gluonts.dataset.artificial import ArtificialDataset
from gluonts.dataset.common import TrainDatasets


def generate_artificial_dataset(
    dataset_path: Path,
    dataset: ArtificialDataset,
    dataset_writer: DatasetWriter,
    prediction_length: Optional[int] = None,
) -> None:
    ds = dataset.generate()
    assert ds.test is not None
    if prediction_length is not None:
        ds.metadata.prediction_length = prediction_length

    dataset = TrainDatasets(metadata=ds.metadata, train=ds.train, test=ds.test)
    dataset.save(
        path_str=str(dataset_path), writer=dataset_writer, overwrite=True
    )
