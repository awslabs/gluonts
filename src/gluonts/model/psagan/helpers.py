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

import logging
import tarfile
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

# import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S]",
)
logger = logging.getLogger(__name__)


def nf(stage, fmap_base=8192, fmap_decay=1.7, fmap_max=32, fmap_min: int = 32):
    """Computes the number of feature map given the current stage.

    This function is inspired from the following Github repo:
    https://github.com/tkarras/progressive_growing_of_gans/blob/master/networks.py

    Arguments:
        stage:
            Integer
    Returns:
        Integer, which is the number of feature map wanted at this stage.
    """
    return min(
        max(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_min), fmap_max
    )


def EMA(loss: np.ndarray, alpha: float = 0.2):
    """
    loss: one dimensional numpy array or of shape (D,1)
    alpha: value to compute the exponential moving average
    """
    loss = np.squeeze(loss)
    try:
        assert len(loss) > 0
    except AssertionError:
        logger.critical("Loss is empty, no EMA computation is done")
        return
    ema_loss = np.empty(len(loss))
    ema_loss[0] = loss[0]
    for index, item in enumerate(loss[1:]):
        ema_loss[index] = (
            alpha * loss[index] + (1 - alpha) * ema_loss[index - 1]
        )

    return ema_loss


def produce_combos(di):
    combos = product(
        *([(key, val) for val in value] for key, value in di.items())
    )
    combos = [dict(combination) for combination in combos]
    return combos


def check_condition(list_conditions, condition):
    for cond in list_conditions:
        if cond.items() <= condition.items():
            return True
    return False


def dict_to_list_of_dict(di):
    combos = product(
        *([(key, val) for val in value] for key, value in di.items())
    )
    combos = [dict(combination) for combination in combos]
    return combos


def look_for_model_simple_dict(
    path_to_csv, path_to_directory, dict_conditions
):
    df = pd.read_csv(path_to_csv)
    filtered_df = df.loc[
        (df[list(dict_conditions)] == pd.Series(dict_conditions)).all(axis=1)
    ]
    list_models = []
    for idx, row in filtered_df.iterrows():
        model_id = row["model_id"]
        use_loss = row["use_loss"]
        target_len = row["target_len"]
        path_to_predictor = (
            Path(path_to_directory)
            / f"synthetic_out_{model_id}"
            / f"{use_loss}_{target_len}"
        )
        if path_to_predictor.exists():
            list_models.append(path_to_predictor)
    return list_models


def look_for_model(path_to_csv, path_to_directory, dict_conditions):
    list_dict = dict_to_list_of_dict(dict_conditions)
    list_models = []
    for di in list_dict:
        list_models += look_for_model_simple_dict(
            path_to_csv, path_to_directory, di
        )
    return list_models


def jobname_to_unzipedname(s):
    l = []
    s = s.split("-")[-3:]
    l.append(s[0][:2] + "_" + s[0][2:4] + "_" + s[0][4:])
    l.append(s[1][:2] + "_" + s[1][2:4] + "_" + s[1][4:])
    l.append(s[2])
    final_s = "__".join(l)
    final_s = "synthetic_out_" + final_s
    return final_s


def unzip_single_files(path_to_zipped_file):
    path_to_zipped_file = Path(path_to_zipped_file)
    save_directory = path_to_zipped_file.parents[1] / "UnzipTrainedModels"
    tar = tarfile.open(path_to_zipped_file, "r:gz")
    tar.extractall(save_directory)
    tar.close()
    logger.info("Succesfully unziped the file")


def unzip_files_in_directory(path_to_directory):
    l = list(Path(path_to_directory).glob("*.tar.gz"))
    for idx, path in enumerate(l):
        parents = path.parents
        jobname = path.name[:-7]
        logger.info(f"{idx+1}/{len(l)}: Looking into {jobname}")
        unziped_name = jobname_to_unzipedname(jobname)
        path_to_zipped_file = parents[1] / "UnzipTrainedModels" / unziped_name
        if not path_to_zipped_file.exists():
            print(path_to_zipped_file.exists())
            logger.info(
                f"Unziping {jobname} and saving it under {path_to_zipped_file}"
            )
            unzip_single_files(path)
        else:
            logger.info(
                f"{jobname} has already been unziped and is under {path_to_zipped_file}"
            )


# unzip_files_in_directory("/home/ec2-user/SageMaker/TrainedModels")


def cardinality_fct(missing_values, cold_start_value):
    if missing_values == 0:
        if cold_start_value == 0:
            di = {
                "m4_hourly": 414,
                "electricity": 321,
                "solar-energy": 137,
                "exchange_rate": 8,
                "traffic": 862,
                "s3://s3-violation-test-bucket/datasets/solar_nips_scaled/": 137,
                "s3://s3-violation-test-bucket/datasets/traffic_nips_scaled/": 963,
                "s3://s3-violation-test-bucket/datasets/m4_hourly_scaled/": 414,
                "s3://s3-violation-test-bucket/datasets/exchange_rate_nips_scaled/": 8,
                "s3://s3-violation-test-bucket/datasets/electricity_nips_scaled/": 370,
            }
            return di
        else:
            di_COLD = {
                "m4_hourly": 414,
                "electricity": 321,
                "solar-energy": 137,
                "exchange_rate": 8,
                "traffic": 862,
                f"s3://s3-violation-test-bucket/datasets_cold/solar_nips_cold_{cold_start_value}/": 137,
                f"s3://s3-violation-test-bucket/datasets_cold/traffic_nips_cold_{cold_start_value}/": 963,
                f"s3://s3-violation-test-bucket/datasets_cold/m4_hourly_cold_{cold_start_value}/": 414,
                f"s3://s3-violation-test-bucket/datasets_cold/exchange_rate_nips_cold_{cold_start_value}/": 8,
                f"s3://s3-violation-test-bucket/datasets_cold/electricity_nips_cold_{cold_start_value}/": 370,
            }
            return di_COLD
    else:
        di = {
            f"s3://s3-violation-test-bucket/mv-datasets2/solar_nips_scaled-stretch-len-{missing_values}/": 137,
            f"s3://s3-violation-test-bucket/mv-datasets2/traffic_nips_scaled-stretch-len-{missing_values}/": 963,
            f"s3://s3-violation-test-bucket/mv-datasets2/m4_hourly_scaled-stretch-len-{missing_values}/": 414,
            f"s3://s3-violation-test-bucket/mv-datasets2/exchange_rate_nips_scaled-stretch-len-{missing_values}/": 8,
            f"s3://s3-violation-test-bucket/mv-datasets2/electricity_nips_scaled-stretch-len-{missing_values}/": 370,
        }
        return di


def nb_features_fct(missing_values, cold_start_value):
    if missing_values == 0:
        if cold_start_value == 0:
            di = {
                "m4_hourly": [4],
                "electricity": [4],
                "solar-energy": [5],
                "exchange_rate": [3],
                "traffic": [4],
                "s3://s3-violation-test-bucket/datasets/solar_nips_scaled/": [
                    4
                ],
                "s3://s3-violation-test-bucket/datasets/traffic_nips_scaled/": [
                    4
                ],
                "s3://s3-violation-test-bucket/datasets/m4_hourly_scaled/": [
                    4
                ],
                "s3://s3-violation-test-bucket/datasets/exchange_rate_nips_scaled/": [
                    3
                ],
                "s3://s3-violation-test-bucket/datasets/electricity_nips_scaled/": [
                    4
                ],
            }
            return di
        else:
            di_COLD = {
                "m4_hourly": [4],
                "electricity": [4],
                "solar-energy": [5],
                "exchange_rate": [3],
                "traffic": [4],
                f"s3://s3-violation-test-bucket/datasets_cold/solar_nips_cold_{cold_start_value}/": [
                    4
                ],
                f"s3://s3-violation-test-bucket/datasets_cold/traffic_nips_cold_{cold_start_value}/": [
                    4
                ],
                f"s3://s3-violation-test-bucket/datasets_cold/m4_hourly_cold_{cold_start_value}/": [
                    4
                ],
                f"s3://s3-violation-test-bucket/datasets_cold/exchange_rate_nips_cold_{cold_start_value}/": [
                    3
                ],
                f"s3://s3-violation-test-bucket/datasets_cold/electricity_nips_cold_{cold_start_value}/": [
                    4
                ],
            }
            return di_COLD

    else:
        di = {
            f"s3://s3-violation-test-bucket/mv-datasets2/solar_nips_scaled-stretch-len-{missing_values}/": [
                4
            ],
            f"s3://s3-violation-test-bucket/mv-datasets2/traffic_nips_scaled-stretch-len-{missing_values}/": [
                4
            ],
            f"s3://s3-violation-test-bucket/mv-datasets2/m4_hourly_scaled-stretch-len-{missing_values}/": [
                4
            ],
            f"s3://s3-violation-test-bucket/mv-datasets2/exchange_rate_nips_scaled-stretch-len-{missing_values}/": [
                3
            ],
            f"s3://s3-violation-test-bucket/mv-datasets2/electricity_nips_scaled-stretch-len-{missing_values}/": [
                4
            ],
        }
        return di


def ds(missing_value: float, cold_start_value: float):

    if missing_value == 0:
        if cold_start_value == 0:
            DS = [
                "s3://s3-violation-test-bucket/datasets/traffic_nips_scaled/",
                "s3://s3-violation-test-bucket/datasets/m4_hourly_scaled/",
                "s3://s3-violation-test-bucket/datasets/solar_nips_scaled/",
                # "s3://s3-violation-test-bucket/datasets/exchange_rate_nips_scaled/",
                "s3://s3-violation-test-bucket/datasets/electricity_nips_scaled/",
            ]
            return DS
        else:
            DS_COLD = [
                f"s3://s3-violation-test-bucket/datasets_cold/traffic_nips_cold_{cold_start_value}/",
                f"s3://s3-violation-test-bucket/datasets_cold/m4_hourly_cold_{cold_start_value}/",
                f"s3://s3-violation-test-bucket/datasets_cold/solar_nips_cold_{cold_start_value}/",
                # "s3://s3-violation-test-bucket/datasets_cold/exchange_rate_nips_cold_{cold_start_value}/",
                f"s3://s3-violation-test-bucket/datasets_cold/electricity_nips_cold_{cold_start_value}/",
            ]
            return DS_COLD
    else:
        DS_MV = [
            f"s3://s3-violation-test-bucket/mv-datasets2/traffic_nips_scaled-stretch-len-{missing_value}/",
            f"s3://s3-violation-test-bucket/mv-datasets2/m4_hourly_scaled-stretch-len-{missing_value}/",
            f"s3://s3-violation-test-bucket/mv-datasets2/solar_nips_scaled-stretch-len-{missing_value}/",
            # "s3://s3-violation-test-bucket/mv-datasets2/exchange_rate_nips_scaled-stretch-len-{missing_value}/",
            f"s3://s3-violation-test-bucket/mv-datasets2/electricity_nips_scaled-stretch-len-{missing_value}/",
        ]
        return DS_MV


def set_context_length(hp: dict, CTX_LEN: int) -> dict:
    activate_context = bool(hp["context_length"][0])
    if activate_context:
        target_len = hp["target_len"][0]
        hp["context_length"] = [min(target_len, CTX_LEN)]
    return hp
