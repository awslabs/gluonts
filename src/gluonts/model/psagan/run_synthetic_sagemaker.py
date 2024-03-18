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


import tempfile
from datetime import datetime
from pathlib import Path
from random import randint
from typing import List, Optional, Tuple

import pandas as pd
import sagemaker
from keymaker import KeyMaker

import gluonts
from gluonts.model.syntheticTransformer._estimator import syntheticEstimator
from gluonts.model.syntheticTransformer._trainer import Trainer
from gluonts.nursery.sagemaker_sdk import GluonTSFramework


def run(
    path_to_pretrain: str,
    num_epochs: int = 5,
    lr_generator: float = 0.0005,
    lr_discriminator: float = 0.0005,
    betas_generator: Tuple[float, float] = (0.9, 0.999),
    betas_discriminator: Tuple[float, float] = (0.9, 0.999),
    schedule: List[int] = None,
    nb_epoch_fade_in_new_layer: int = 100,
    nb_step_discrim: int = 2,
    EMA_value: float = 0.2,
    batch_size: int = 64,
    num_batches_per_epoch: int = 8,
    target_len: int = 2 ** 5,
    nb_features: int = 4,
    ks_conv: int = 3,
    key_features: int = 10,
    value_features: int = 10,
    ks_value: int = 1,
    ks_query: int = 1,
    ks_key: int = 1,
    ds: str = "m4_hourly",
    num_workers: int = 0,
    device: str = "cpu",
    use_loss: str = "hinge",
    momment_loss: float = 0.0,
    scaling_penalty: float = 0.0,
    scaling: str = "local",
    cardinality: Optional[List[int]] = None,
    embedding_dim: int = 10,
    self_attention: bool = True,
    channel_nb: int = 32,
    encoder_network_factor: float = None,
    pos_enc_dimension: int = 10,
    path_to_csv: str = None,
    context_length: int = 0,
    LARS: bool = False,
    missing_values_stretch: int = 0,
    cold_start_value: float = 0,
):
    km = KeyMaker()
    region_name = "us-east-1"
    bucket_name = "s3://s3-violation-test-bucket"  # Change to your bucket
    iam_role = "arn:aws:iam::670864377759:role/service-role/AmazonSageMaker-ExecutionRole-20181125T162939"

    print(f"bucket_name = '{bucket_name}'")
    print(f"region_name = '{region_name}'")
    print(f"iam_role = '{iam_role}'")

    sagemaker_session = sagemaker.Session(
        boto_session=km.profile("mlf-bench", region=region_name)
    )

    # general_instance_type = "cpu"
    general_instance_type = device  # alternative

    instance_type = (
        "ml.c5.4xlarge" if general_instance_type == "cpu" else "ml.p2.xlarge"
    )

    if general_instance_type == "cpu":
        # docker_image = "670864377759.dkr.ecr.us-west-2.amazonaws.com/gluonts/sagemaker:latest"
        docker_image = f"763104351884.dkr.ecr.{region_name}.amazonaws.com/mxnet-training:1.6.0-cpu-py36-ubuntu16.04"
    else:
        docker_image = f"763104351884.dkr.ecr.{region_name}.amazonaws.com/mxnet-training:1.6.0-gpu-py36-cu101-ubuntu16.04"
    print(f"docker_image = '{docker_image}'")

    def ds_to_ds(dataset_path):
        if "m4" in dataset_path:
            return "m4_hourly"
        elif "solar" in dataset_path:
            return "solar-energy"
        elif "electricity" in dataset_path:
            return "electricity"
        elif "exchange" in dataset_path:
            return "electricity_rate"
        elif "traffic" in dataset_path:
            return "traffic"

    ds_to_encoder = {
        "m4_hourly": "./gluonts/model/syntheticTransformer/CNN_embedder/m4_hourly",
        "electricity": "./gluonts/model/syntheticTransformer/CNN_embedder/electricity",
        "solar-energy": "./gluonts/model/syntheticTransformer/CNN_embedder/solar-energy",
        "exchange_rate": "./gluonts/model/syntheticTransformer/CNN_embedder/exchange_rate",
        "traffic": "./gluonts/model/syntheticTransformer/CNN_embedder/traffic",
        "s3://s3-violation-test-bucket/datasets/solar_nips_scaled/": "./gluonts/model/syntheticTransformer/CNN_embedder/solar-energy",
        "s3://s3-violation-test-bucket/datasets/traffic_nips_scaled/": "./gluonts/model/syntheticTransformer/CNN_embedder/traffic",
        "s3://s3-violation-test-bucket/datasets/m4_hourly_scaled/": "./gluonts/model/syntheticTransformer/CNN_embedder/m4_hourly",
        "s3://s3-violation-test-bucket/datasets/exchange_rate_nips_scaled/": "./gluonts/model/syntheticTransformer/CNN_embedder/exchange_rate",
        "s3://s3-violation-test-bucket/datasets/electricity_nips_scaled/": "./gluonts/model/syntheticTransformer/CNN_embedder/electricity",
    }

    now = datetime.now()
    now_id = now.strftime("%d_%m_%Y__%H_%M_%S")
    now_title = now.strftime("%d%m%Y-%H%M%S")
    RANDOM = randint(0, 1000)

    if use_loss == "lsgan":
        title_loss = "PSA-GAN"
    elif use_loss == "wgan":
        title_loss = "PSA-GAN-W"

    base_job_description = f"{title_loss}-{ds_to_ds(ds)}-len{target_len}-{now_title}-{RANDOM}".replace(
        "_", "-"
    ).replace(
        ".", ""
    )

    dataset_name = ds
    # dataset_name = "s3://<your-custom-dataset-location>" # if using a custom dataset

    # only using temporary directory for demonstration
    temp_dir = tempfile.TemporaryDirectory()
    temp_dir_path = Path(temp_dir.name)

    requirements_dot_txt_file_name = "requirements.txt"
    requirements_dot_txt_file_content = "torch \n holidays>=0.9 \n matplotlib~=3.0 \n numpy~=1.16 \n pandas>=1.0 \n pydantic~=1.1,<1.7 \n tqdm~=4.23 \n toolz~=0.10 \n ujson~=1.35"

    # create the requirements.txt file
    with open(
        temp_dir_path / requirements_dot_txt_file_name, "w"
    ) as req_file:  # has to be called requirements.txt
        req_file.write(requirements_dot_txt_file_content)
    my_requirements_txt_file_path = str(
        temp_dir_path / requirements_dot_txt_file_name
    )
    print(f"my_requirements_txt_file_path = '{my_requirements_txt_file_path}'")

    if encoder_network_factor == 0:
        encoder_network_path = None
        encoder_network_factor = None
    else:
        encoder_network_path = ds_to_encoder[ds_to_ds(ds)]

    if "electricity" in ds:
        exclude_index = [322]
    else:
        exclude_index = None

    estimator = syntheticEstimator(
        trainer=Trainer(
            num_epochs=num_epochs,
            lr_generator=lr_generator,
            lr_discriminator=lr_discriminator,
            schedule=schedule,
            nb_step_discrim=nb_step_discrim,
            nb_epoch_fade_in_new_layer=nb_epoch_fade_in_new_layer,
            EMA_value=EMA_value,
            device=device,
            use_loss=use_loss,
            momment_loss=momment_loss,
            scaling_penalty=scaling_penalty,
            betas_generator=betas_generator,
            betas_discriminator=betas_discriminator,
            scaling=scaling,
            encoder_network_path=encoder_network_path,
            encoder_network_factor=encoder_network_factor,
            LARS=LARS,
        ),
        freq="random_string",
        batch_size=batch_size,
        num_batches_per_epoch=num_batches_per_epoch,
        target_len=target_len,
        nb_features=nb_features,
        ks_conv=ks_conv,
        key_features=key_features,
        value_features=value_features,
        ks_value=ks_value,
        ks_query=ks_query,
        ks_key=ks_key,
        num_workers=num_workers,
        device=device,
        path_to_pretrain=path_to_pretrain,
        scaling=scaling,
        cardinality=cardinality,
        embedding_dim=embedding_dim,
        self_attention=self_attention,
        channel_nb=channel_nb,
        pos_enc_dimension=pos_enc_dimension,
        context_length=context_length,
        exclude_index=exclude_index,
    )
    di = {
        "num_epochs": num_epochs,
        "lr_generator": lr_generator,
        "lr_discriminator": lr_discriminator,
        "schedule": schedule,
        "nb_step_discrim": nb_step_discrim,
        "nb_epoch_fade_in_new_layer": nb_epoch_fade_in_new_layer,
        "EMA_value": EMA_value,
        "device": device,
        "use_loss": use_loss,
        "momment_loss": momment_loss,
        "scaling_penalty": scaling_penalty,
        "betas_generator": betas_generator,
        "betas_discriminator": betas_discriminator,
        "scaling": scaling,
        "batch_size": batch_size,
        "num_batches_per_epoch": num_batches_per_epoch,
        "target_len": target_len,
        "nb_features": nb_features,
        "ks_conv": ks_conv,
        "key_features": key_features,
        "value_features": value_features,
        "ks_value": ks_value,
        "ks_query": ks_query,
        "ks_key": ks_key,
        "num_workers": num_workers,
        "path_to_pretrain": path_to_pretrain,
        "cardinality": cardinality,
        "embedding_dim": embedding_dim,
        "dataset": ds_to_ds(dataset_name),
        "which_model": "PSA-GAN" if use_loss == "lsgan" else "PSA-GAN-W",
        "model_id": f"{now_id}__{RANDOM}",
        "bucket_name": bucket_name,
        "cardinality": cardinality,
        "channel_nb": channel_nb,
        "pos_enc_dimension": pos_enc_dimension,
        "self_attention": self_attention,
        "TrainingJobName": base_job_description,
        "TrainingJobStatus": "InProgress",
        "DownloadStatus": False,
        "encoder_network_factor": encoder_network_factor,
        "context_length": context_length,
        "LARS": LARS,
        "UploadS3": False,
        "UploadS3Location": False,
        "SaveLocation": False,
        "missing_values_stretch": missing_values_stretch,
        "cold_start_value": cold_start_value,
    }
    print(di)
    df = pd.read_csv(path_to_csv, dtype="object")
    df = df.append(di, ignore_index=True)
    df.to_csv(path_to_csv, index=False)

    print(gluonts.__path__)
    my_experiment = GluonTSFramework(
        sagemaker_session=sagemaker_session,
        role=iam_role,
        image_uri=docker_image,
        base_job_name=base_job_description,
        instance_type=instance_type,
        dependencies=[gluonts.__path__[0], my_requirements_txt_file_path],
        output_path=bucket_name,  # experiment_parent_dir, # optional, but recommended
        code_location=bucket_name,  # optional, but recommended
        entry_point=f"{gluonts.__path__[0]}/model/syntheticTransformer/synthtetic_entry_point.py",
        hyperparameters={"model_id": f"{now_id}__{RANDOM}", "config_dict": di},
    )

    my_experiment.train(
        dataset=dataset_name,
        estimator=estimator,
        wait=False,
        job_name=base_job_description,
    )
