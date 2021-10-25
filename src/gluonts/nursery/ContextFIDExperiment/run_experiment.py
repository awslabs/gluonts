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
from datetime import datetime
from pathlib import Path
from random import choice

import pandas as pd
import torch
import yaml
from icecream import ic

from gluonts.model.psagan.cnn_encoder._model import CausalCNNEncoder
from gluonts.model.psagan.helpers import look_for_model
from gluonts.nursery.ContextFIDExperiment.experimental_setup_refactored import (
    Experiment,
)
from gluonts.nursery.ContextFIDExperiment.res2tex import toLatex

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S]",
)
logger = logging.getLogger(__name__)


def main(
    embed_dir: str,
    gan_model_path: str,
    csv_path_save_result: Path = None,
    csv_path_experiment: Path = None,
    nb_run_experiment: int = 10,
    save_location: Path = Path.cwd(),
):

    data_model = yaml.load(
        open(Path(gan_model_path) / "data.yml"), Loader=yaml.FullLoader
    )
    TrainingJobName = data_model["TrainingJobName"]
    df_experiment = pd.read_csv(csv_path_experiment)
    _, di = (
        df_experiment[df_experiment["TrainingJobName"] == TrainingJobName]
        .to_dict(orient="index")
        .popitem()
    )
    embed_dir = Path(embed_dir) / data_model["dataset"]
    embed_model_path = list(
        list(embed_dir.glob("model_*"))[0].glob("model_*")
    )[0]
    embed_config_path = list(
        list(embed_dir.glob("out_*"))[0].glob("configuration.txt")
    )[0]

    latex_obj = toLatex(path_to_config_file_CNN=embed_config_path)

    embedder = CausalCNNEncoder(
        in_channels=latex_obj._get_in_channels_CNN(),
        channels=latex_obj._get_channels_CNN(),
        depth=latex_obj._get_depth_CNN(),
        reduced_size=latex_obj._get_reduced_size_CNN(),
        out_channels=latex_obj._get_out_channels_CNN(),
        kernel_size=latex_obj._get_kernel_size_CNN(),
    )
    print("HELLO", embed_model_path)
    embedder.load_state_dict(
        torch.load(str(embed_model_path), map_location=torch.device("cpu"))
    )
    embedder.eval()

    exp = Experiment(
        model_similarity_score=embedder,
        gan_model_path=gan_model_path,
    )
    fid_ts, fid_embed = exp.run_FID(nb_run_experiment)

    di["FID_MEAN"] = torch.mean(fid_embed).item()
    di["FID_STD"] = torch.std(fid_embed).item()
    di["FID_TS_MEAN"] = torch.mean(fid_ts).item()
    di["FID_TS_STD"] = torch.std(fid_ts).item()
    di["nb_run_experiment"] = nb_run_experiment
    if csv_path_save_result is not None:
        df = pd.read_csv(csv_path_save_result)
        existing_id = list(df["ID_EXPERIMENT"])
        now = datetime.now()
        now_id = now.strftime("%d_%m_%Y__%H_%M_%S")
        random_id = choice(
            [ele for ele in range(0, 10000000) if ele not in existing_id]
        )
        di["ID_EXPERIMENT"] = f"{now_id}___{random_id}"
        df = df.append(di, ignore_index=True)
        df.to_csv(csv_path_save_result, index=False)


df_train = pd.read_csv(
    "./gluonts/model/suntheticTransformer/TrainingFollowUp.csv"
)
df_exp = pd.read_csv(
    "./gluonts/model/suntheticTransformer/ExperimentFollowUp.csv"
)

df_unexperimented = df_train[
    ~df_train.TrainingJobName.isin(df_exp.TrainingJobName)
]
d = df_unexperimented[
    (df_unexperimented["target_len"].isin([16, 32, 64, 128, 256]))
    # &(df_unexperimented["cold_start_value"].isin([0., None]))
    # &(df_unexperimented["missing_values_stretch"].isin([0., None]))
    & (df_unexperimented["DownloadStatus"] == True)
    & (df_unexperimented["TrainingJobStatus"] == "Completed")
]

di = {"TrainingJobName": list(d["TrainingJobName"])}

list_models = look_for_model(
    Path(
        "/home/ec2-user/SageMaker/gluon-ts-gan/src/gluonts/model/syntheticTransformer/TrainingFollowUp.csv"
    ),
    Path("/home/ec2-user/SageMaker/UnzipTrainedModels"),
    di,
)
logger.info(f"Here is the list of models to run : {list_models}")
answer = input(
    f"You are going to run {len(list_models)} experiments, are you sure ? (yes or no): "
)
if answer == "yes":
    # pass
    for model_path in list_models:
        main(
            embed_dir="/home/ec2-user/SageMaker/gluon-ts-gan/src/gluonts/model/syntheticTransformer/CNN_embedder",
            csv_path_save_result="/home/ec2-user/SageMaker/gluon-ts-gan/src/gluonts/model/syntheticTransformer/ExperimentFollowUp.csv",
            csv_path_experiment="/home/ec2-user/SageMaker/gluon-ts-gan/src/gluonts/model/syntheticTransformer/TrainingFollowUp.csv",
            gan_model_path=model_path,
            nb_run_experiment=10,
            # save_location=Path("/Users/pauljeha/Desktop/FID scores")
        )
else:
    print("Ok, not running experiments.")
