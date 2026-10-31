# Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import os
from pathlib import Path

from typing import Iterable, List

from pathlib import Path, PosixPath
import shutil

import itertools

import json

import numpy as np

import pandas as pd

# import openpyxl as op


def rm_file_or_dir(file_path: str):
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path, ignore_errors=True)
    except Exception as e:
        print("Failed to delete %s. Reason: %s" % (file_path, e))


def clear_dir(path: str):
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        rm_file_or_dir(file_path)


def save_args(args: dict, path: PosixPath):
    args_dict = args.copy()

    args_dict = {
        key: (str(value) if isinstance(value, Path) else value) for key, value in args_dict.items()
    }

    with open(path, "w") as fp:
        json.dump(args_dict, fp, sort_keys=True, indent=4)


def take_n_cycle(it: Iterable, n: int) -> List:
    cycle_it = itertools.cycle(it)
    return list(itertools.islice(cycle_it, n))


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


# def read_xlsx(filename, sheet_num: int = 0, header: bool = True, skiprows: Optional[int] = None, nrows: Optional[int] = None):
#     wb = op.load_workbook(filename=filename, read_only=True)
#     ws = wb.worksheets[sheet_num]
#     if skiprows is None:
#         skiprows = 0
#     if nrows is None:
#         nrows = ws.max_row-skiprows
#         rows = ws.iter_rows(min_row=skiprows+1, max_row=skiprows+nrows)
#     # Get headers
#     if header:
#         headers = [cell.value for cell in next(rows)]
#     # Load the data
#     data = []
#     for row in rows:
#         data.append( [cell.value for cell in row] )
#     # Convert to a df
#     df = pd.DataFrame(data, columns=headers)
#     return df
