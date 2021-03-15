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


from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Dict, Tuple, List, Union
from collections import OrderedDict
from pathlib import Path
from itertools import product
from functools import reduce
import traceback
import textwrap
import socket
import json

import torch as pt
import numpy as np
import pandas as pd

from .distributed import is_main_process, synchronize

if TYPE_CHECKING:
    from ..dataset import MetaDataset


class HyperOptManager(object):
    def __init__(
        self,
        dataset: MetaDataset,
        work_dir: Path,
        fixed_params: Dict,
        cuda_device: int,
        key_factor: str = "loss",
        min_mode: bool = True,
        resume: bool = False,
    ):
        self.dataset = dataset
        self.param_names, self.varied_params = zip(
            *sorted(dataset.hyperparam_space.items(), key=lambda x: x[0])
        )
        self.fixed_params = fixed_params
        self.work_dir = work_dir
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.cuda_device = cuda_device
        self.key_factor = key_factor
        self.min_mode = min_mode

        self.exp_ids = OrderedDict()
        self.records = OrderedDict()
        if resume:
            self.load_records()

    def __getitem__(self, key: Union[Tuple, int]) -> Union[int, Tuple]:
        if isinstance(key, tuple):
            index = 0
            for i, param in enumerate(key):
                idx = self.varied_params[i].index(param)
                index = index * len(self.varied_params[i]) + idx
            return index
        else:
            params = []
            for vp in reversed(self.varied_params):
                idx = key % len(vp)
                params.insert(0, vp[idx])
                key = key // len(vp)
            return tuple(params)

    def __len__(self):
        return reduce(lambda p, vp: p * len(vp), self.varied_params, 1)

    def hyperparameters(self, n_iter: int):
        n_choices_per_param = [len(vp) for vp in self.varied_params]
        n_choices = reduce(lambda p, x: p * x, n_choices_per_param, 1)
        n_covered = len(self.records)
        if n_iter > n_choices - n_covered:
            raise ValueError(
                f"# iterations {n_iter} exceeds the remaining grids {n_choices-n_covered}"
            )
        samples = list(np.random.choice(n_choices, n_iter + n_covered))
        for params in self.records:
            index = self[params]
            if index in samples:
                samples.remove(index)
        samples = samples[:n_iter]
        for index in samples:
            yield self[index]

    def get_log_dir(self, params) -> str:
        exp_id = f"{self[params]:09d}-{socket.gethostname()}"
        self.exp_ids[params] = exp_id
        log_dir = self.work_dir.joinpath(exp_id)
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir

    def get_optimal_model(self) -> Path:
        if len(self.records) == 0:
            raise ValueError("No model has been trained!")
        selector = np.nanargmin if self.min_mode else np.nanargmax
        models = list(self.records.keys())
        values = [r[self.key_factor] for r in self.records.values()]
        dirname = self.exp_ids[models[selector(values)]]
        return self.work_dir.joinpath(dirname)

    def dump_records(self):
        columns = list(self.param_names)
        index = []
        data = []
        for params, record in self.records.items():
            values = list(params)
            for name, value in record.items():
                if name not in columns:
                    columns.append(name)
                values.append(value)
            data.append(values)
            index.append(self.exp_ids[params])
        df = pd.DataFrame(data, index=index, columns=columns)
        df.to_csv(self.work_dir.joinpath("records.csv"))

    def load_records(self):
        print(f"Loading records from {self.work_dir}")
        record_path = self.work_dir.joinpath("records.csv")
        if record_path.exists():
            df = pd.read_csv(record_path, index_col=0)
            for exp_id, record in df.iterrows():
                params = (
                    record.loc[list(self.param_names)]
                    .values.flatten()
                    .tolist()
                )
                params = tuple(
                    eval(x) if isinstance(x, str) else x for x in params
                )
                self.exp_ids[params] = exp_id
                self.records[params] = record.iloc[
                    len(self.param_names) :
                ].to_dict()
            return
        raise RuntimeError("Cannot load previous checkpoint in random search.")

    def update_record(self, params: Tuple, **record):
        if self.key_factor not in record:
            raise ValueError(f"key factor {self.key_factor} is not provided.")
        self.records[params] = record
        self.dump_records()

    @staticmethod
    def print_params(params: Dict):
        print("Current Hyperparams:")
        for k, v in params.items():
            print(f"\t{k:>16}{str(v):>16}")

    def run_training(self, params: Dict) -> Dict:
        raise NotImplementedError

    def run_test(self, exp_dir: Path):
        raise NotImplementedError

    def random_search(self, n_iter: int):
        for params in self.hyperparameters(n_iter):
            configs = dict(zip(self.param_names, params))
            self.print_params(configs)
            configs.update(self.fixed_params)
            log_dir = self.get_log_dir(params)
            if is_main_process():
                with log_dir.joinpath("configs.json").open("w") as f:
                    json.dump(configs, f, indent=4)
            configs["cuda_device"] = self.cuda_device
            configs["log_dir"] = log_dir
            try:
                exp_info = self.run_training(configs)
            except Exception as e:
                print(
                    f"Current config has problem:\n{textwrap.indent(traceback.format_exc(), ' '*4)}skipped."
                )
                continue
            self.update_record(params, **exp_info)
        if is_main_process():
            exp_dir = self.get_optimal_model()
            test_info = self.run_test(exp_dir)
        else:
            test_info = None
        synchronize()
        return test_info

    def fixed_train(self):
        exp_id = f"fixed-{socket.gethostname()}"
        log_dir = self.work_dir.joinpath(exp_id)
        log_dir.mkdir(parents=True, exist_ok=True)
        configs = self.dataset.fixed_hyperparams
        configs.update(self.fixed_params)

        if is_main_process():
            with log_dir.joinpath("configs.json").open("w") as f:
                json.dump(configs, f, indent=4)
        configs["cuda_device"] = self.cuda_device
        configs["log_dir"] = log_dir
        try:
            _ = self.run_training(configs)
        except Exception as e:
            print(
                f"Current config has problem:\n{textwrap.indent(traceback.format_exc(), ' '*4)}skipped."
            )
            return
        test_info = self.run_test(log_dir)
        return test_info
