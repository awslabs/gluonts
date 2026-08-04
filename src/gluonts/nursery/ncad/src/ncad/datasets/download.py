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
from pathlib import Path, PosixPath

from typing import List, Union

import tarfile

from ncad.utils import rm_file_or_dir


def download(
    data_dir: Union[str, PosixPath] = "~/ncad_datasets",
    benchmarks: Union[str, List[str]] = ["kpi", "nasa", "smd", "swat", "yahoo"],
    swat_path: Union[str, PosixPath] = None,
    yahoo_path: Union[str, PosixPath] = None,
) -> None:
    """Download benchmark datasets for Anomaly detection on Time Series.

    Args:
        data_dir : Directory to store the data files
        benchmarks : List of benchmarks to be downloaded.
            Currently, the following datasets are supported:
            ['kpi','nasa','smd','swat','yahoo']
            If 'swat' is included, swat_path is required.
            If 'yahoo' is included, yahoo_path is required.
        swat_path : Path to the file provided by iTrust.
        yahoo_path : Path to the tar file downloaded from Yahoo Labs (only used if 'yahoo' is in benchmarks).

    Sources:
        https://github.com/khundman/telemanom
        https://github.com/NetManAIOps/KPI-Anomaly-Detection
    """

    # Transform data_dir to Path
    data_dir = PosixPath(data_dir).expanduser() if str(data_dir).startswith("~") else Path(data_dir)

    if isinstance(benchmarks, str):
        benchmarks = [
            benchmarks,
        ]

    # Create directory
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if "kpi" in benchmarks:
        print("Preparing KPI dataset...")
        cmd_string = (
            f"cd {data_dir}"
            + " && rm -rf kpi"
            + " && git clone https://github.com/NetManAIOps/KPI-Anomaly-Detection.git kpi"
            + " && cd kpi"
            + " && unzip Finals_dataset/phase2_ground_truth.hdf.zip"
            + " && unzip Finals_dataset/phase2_train.csv.zip"
        )
        os.system(cmd_string)
        # remove unnecessary files
        kpi_files = ["phase2_ground_truth.hdf", "phase2_train.csv"]
        rm_these = set(os.listdir(data_dir / "kpi")) - set(kpi_files)
        for file in rm_these:
            rm_file_or_dir(data_dir / "kpi" / file)
        print("... KPI dataset ready!")

    if "nasa" in benchmarks:
        print("Preparing NASA datasets (SMAP and MSL)...")
        cmd_string = (
            f"cd {data_dir}"
            + " && rm -rf nasa"
            + " && curl -O https://s3-us-west-2.amazonaws.com/telemanom/data.zip"
            + " && unzip data.zip && rm data.zip"
            + " && mv data nasa"
            + " && wget https://raw.githubusercontent.com/khundman/telemanom/master/labeled_anomalies.csv -P nasa"
        )
        os.system(cmd_string)
        # remove unnecessary files
        nasa_files = ["train", "test", "labeled_anomalies.csv"]
        rm_these = set(os.listdir(data_dir / "nasa")) - set(nasa_files)
        for file in rm_these:
            rm_file_or_dir(data_dir / "nasa" / file)
        print("... NASA dataset ready")

    if "smd" in benchmarks:
        print("Preparing SMD dataset...")
        cmd_string = (
            f"cd {data_dir}"
            + " && rm -rf smd"
            + " && rm -rf Omni"
            + " && git clone https://github.com/NetManAIOps/OmniAnomaly.git Omni"
            + " && mv Omni/ServerMachineDataset smd"
            + " && rm -rf Omni"
        )
        os.system(cmd_string)
        print("... SMD dataset ready")

    if "swat" in benchmarks:
        os.makedirs(data_dir / "swat", exist_ok=True)
        files_swat = ["SWaT_Dataset_Normal_v0.csv", "SWaT_Dataset_Attack_v0.csv"]
        print(
            f"Request SWAT dataset from the iTrust Labs (https://itrust.sutd.edu.sg/ and copy the files\n{files_swat}\n from Dec2015 in {data_dir/'swat'}"
        )

    if "yahoo" in benchmarks:
        if yahoo_path is None:
            raise ValueError(f"yahoo_path must be provided if 'yahoo' is in benchmarks")
        else:
            print("Preparing Yahoo dataset...")

            yahoo_path = (
                PosixPath(yahoo_path).expanduser()
                if str(yahoo_path).startswith("~")
                else Path(yahoo_path)
            )
            with tarfile.open(str(yahoo_path)) as tar:
                tar.extractall(path=data_dir / "yahoo")
            aux_dir = os.listdir(data_dir / "yahoo")[0]

            yahoo_files = [f"A{i}Benchmark" for i in range(1, 5)]
            cmd_string = f"cd {data_dir}/yahoo"
            for dir_i in yahoo_files:
                cmd_string += f" && mv {aux_dir}/{dir_i} ."
            os.system(cmd_string)
            rm_these = set(os.listdir(data_dir / "yahoo")) - set(yahoo_files)
            for file in rm_these:
                rm_file_or_dir(data_dir / "yahoo" / file)
            print("...Yahoo dataset ready!")
