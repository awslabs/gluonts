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

# Standard library imports

# NOTE: There is no real estimator class for PyTorch networks yet. I'm just
# factoring out transformation and training loop functions to this module
# for now.


import tempfile
from pathlib import Path

import sagemaker
from keymaker import KeyMaker

import gluonts
from gluonts.model.psagan.cnn_encoder._estimator import (
    CausalCNNEncoderEstimator,
)
from gluonts.model.psagan.cnn_encoder._trainer import Trainer
from gluonts.nursery.sagemaker_sdk import GluonTSFramework

# if __name__ == "__main__":


def run(
    num_epochs: int = 800,
    lr: float = 0.0005,
    save_display_frq: int = 1,
    batch_size: int = 64,
    nb_features: int = 1,
    nb_channels: int = 10,
    depth: int = 3,
    reduced_size: int = 160,
    size_embedding: int = 80,
    kernel_size: int = 3,
    subseries_length: int = 30,
    context_length: int = 250,
    max_len: int = 250,
    nb_negative_samples: int = 10,
    device: str = "cpu",
    num_workers: int = 4,
    dataset_name: str = "m4_hourly",
    scaling: str = "global",
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

    general_instance_type = device

    instance_type = (
        "ml.c5.4xlarge" if general_instance_type == "cpu" else "ml.p2.xlarge"
    )

    if general_instance_type == "cpu":
        # docker_image = "670864377759.dkr.ecr.us-west-2.amazonaws.com/gluonts/sagemaker:latest"
        docker_image = f"763104351884.dkr.ecr.{region_name}.amazonaws.com/mxnet-training:1.6.0-cpu-py36-ubuntu16.04"
    else:
        docker_image = f"763104351884.dkr.ecr.{region_name}.amazonaws.com/mxnet-training:1.6.0-gpu-py36-cu101-ubuntu16.04"
    print(f"docker_image = '{docker_image}'")

    base_job_description = "causalCNN"

    # dataset_name = "s3://<your-custom-dataset-location>" # if using a custom dataset

    # only using temporary directory for demonstration
    temp_dir = tempfile.TemporaryDirectory()
    temp_dir_path = Path(temp_dir.name)

    requirements_dot_txt_file_name = "requirements.txt"
    requirements_dot_txt_file_content = (
        "git+https://github.com/awslabs/gluon-ts.git \n torch \n matplotlib"
    )

    # create the requirements.txt file
    with open(
        temp_dir_path / requirements_dot_txt_file_name, "w"
    ) as req_file:  # has to be called requirements.txt
        req_file.write(requirements_dot_txt_file_content)
    my_requirements_txt_file_path = str(
        temp_dir_path / requirements_dot_txt_file_name
    )
    print(f"my_requirements_txt_file_path = '{my_requirements_txt_file_path}'")

    estimator = CausalCNNEncoderEstimator(
        trainer=Trainer(
            num_epochs=num_epochs,
            lr=lr,
            save_display_frq=save_display_frq,
            device=device,
        ),
        freq="random_string",
        batch_size=batch_size,
        nb_features=nb_features,
        nb_channels=nb_channels,
        depth=depth,
        reduced_size=reduced_size,
        size_embedding=size_embedding,
        kernel_size=kernel_size,
        subseries_length=subseries_length,
        context_length=context_length,
        max_len=max_len,
        nb_negative_samples=nb_negative_samples,
        use_feat_dynamic_real=True,
        use_feat_static_cat=True,
        use_feat_static_real=True,
        device=device,
        num_workers=num_workers,
        scaling=scaling,
    )

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
        entry_point=f"{gluonts.__path__[0]}/model/CausalCNNEncoder/causalCNN_entry_point.py",
    )

    my_experiment.train(dataset=dataset_name, estimator=estimator, wait=False)
