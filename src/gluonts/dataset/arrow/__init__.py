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


"""
Arrow Dataset
~~~~~~~~~~~~~

Fast and efficient datasets using `pyarrow`.

This module provides three file-types:

    * ``ArrowFile`` (arrow random-access binary format)
    * ``ArrowStreamFile`` (arrow streaming binary format)
    * ``ParquetFile``

"""

__all__ = [
    "write_dataset",
    "File",
    "ArrowFile",
    "ArrowStreamFile",
    "ParquetFile",
    "ArrowWriter",
    "ParquetWriter",
]

from .enc import write_dataset, ArrowWriter, ParquetWriter
from .file import File, ArrowFile, ArrowStreamFile, ParquetFile
