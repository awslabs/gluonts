# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
"""Placeholder docstring"""
from __future__ import absolute_import

# Standard library imports
import logging
from pathlib import Path
import os
import time
import json

# Third-party imports
from sagemaker.estimator import Framework
from sagemaker.fw_utils import empty_framework_version_warning
from sagemaker.vpc_utils import VPC_CONFIG_DEFAULT
from  sagemaker import session
import boto3
import s3fs

# First-party imports
from gluonts.sagemaker.defaults import GLUONTS_VERSION
from gluonts.sagemaker.model import GluonTSModel
from gluonts.core import serde
from gluonts.model.estimator import GluonEstimator
from gluonts.dataset import repository

logger = logging.getLogger("sagemaker")

class GluonTSFramework(Framework):
    """Handle end-to-end training and deployment of custom GluonTS code."""

    __framework_name__ = "gluonts"

    LATEST_VERSION = "0.3.3"
    # """The latest version of GluonTS included in the pre-built Docker images."""

    SAGEMAKER_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
    TRAIN_ENTRY_POINT_SCRIPT = str(SAGEMAKER_DIR / "entry_point_scripts" / "train_entry_point.py")
    RUN_ENTRY_POINT_SCRIPT = str(SAGEMAKER_DIR / "entry_point_scripts" / "run_entry_point.py")

    # Hyper parameters for entry points
    DATASET = "DATASET"
    #DATASET_ON_S3 = "DATASET_ON_S3"

    def __init__(
            self,
            entry_point,
            source_dir=None,
            hyperparameters=None,
            framework_version=None,
            image_name=None,
            **kwargs
            # TODO many more missing, like: session, role, base_job_name, train_instance_type, train_instance_count,  metric_definitions, tags, maybe even imge
    ):
        """This ``Estimator`` executes an MXNet script in a managed MXNet
        execution environment, within a SageMaker Training Job. The managed
        MXNet environment is an Amazon-built Docker container that executes
        functions defined in the supplied ``entry_point`` Python script.
        Training is started by calling
        :meth:`~sagemaker.amazon.estimator.Framework.fit` on this Estimator.
        After training is complete, calling
        :meth:`~sagemaker.amazon.estimator.Framework.deploy` creates a hosted
        SageMaker endpoint and returns an
        :class:`~gluonts.sagemaker.GluonTSPredictor` instance that can
        be used to perform inference against the hosted model.
        #TODO add additional functions
        Technical documentation on preparing GluonTSFramework scripts for SageMaker
        training and using the GluonTsFramework Estimator is available on the project
        home-page: #TODO where? https://github.com/aws/sagemaker-python-sdk
        #TODO update description above...

        Args:
            entry_point (str): Path (absolute or relative) to the Python source
                file which should be executed as the entry point. Only needs to be
                specified if you want to use the generic run() method.
                The script needs to be compatible with Python 3.6.
            source_dir (str): Path (absolute or relative) to a directory with
                any other training source code dependencies aside from tne entry
                point file (default: None). Structure within this directory are
                preserved when training on Amazon SageMaker.
            hyperparameters (dict): Hyperparameters that will be used for
                training (default: None). The hyperparameters are made
                accessible as a dict[str, str] to the training code on
                SageMaker. For convenience, this accepts other types for keys
                and values, but ``str()`` will be called to convert them before
                training.
            framework_version (str): MXNet version you want to use for executing
                your model training code. List of supported versions
                #TODO link where to find supported versions.
                If not specified, this will default to 0.3.3.
            image_name (str): If specified, the estimator will use this image for training and
                hosting, instead of selecting the appropriate GluonTS official image based on
                framework_version. It can be an ECR url or dockerhub image and tag.
                Examples:
                    123.dkr.ecr.us-west-2.amazonaws.com/my-custom-image:1.0
                    custom-image:latest.
            **kwargs: Additional kwargs passed to the
                :class:`~sagemaker.estimator.Framework` constructor.
        """
        if framework_version is None:
            logger.warning(empty_framework_version_warning(GLUONTS_VERSION, self.LATEST_VERSION))
        self.framework_version = framework_version or GLUONTS_VERSION

        # TODO: use pre built image if applicable

        super(GluonTSFramework, self).__init__(
            entry_point, source_dir, hyperparameters, image_name=image_name, **kwargs
        )

        # necessary to configure still...
        self.py_version = "py3"

    def create_model(
            self,
            model_server_workers=None,
            role=None,
            vpc_config_override=VPC_CONFIG_DEFAULT,
            entry_point=None,
            source_dir=None,
            dependencies=None,
            image_name=None,
            **kwargs
    ):
        """Create a ``GluonTSModel`` object that can be deployed to an
        ``Endpoint``.
        Args:
            model_server_workers (int): Optional. The number of worker processes
                used by the inference server. If None, server will use one
                worker per vCPU.
            role (str): The ``ExecutionRoleArn`` IAM Role ARN for the ``Model``,
                which is also used during transform jobs. If not specified, the
                role from the Estimator will be used.
            vpc_config_override (dict[str, list[str]]): Optional override for VpcConfig set on
                the model. Default: use subnets and security groups from this Estimator.
                * 'Subnets' (list[str]): List of subnet ids.
                * 'SecurityGroupIds' (list[str]): List of security group ids.
            entry_point (str): Path (absolute or relative) to the local Python source file which
                should be executed as the entry point to training. If not specified, the training
                entry point is used.
            source_dir (str): Path (absolute or relative) to a directory with any other serving
                source code dependencies aside from the entry point file.
                If not specified, the model source directory from training is used.
            dependencies (list[str]): A list of paths to directories (absolute or relative) with
                any additional libraries that will be exported to the container.
                If not specified, the dependencies from training are used.
            image_name (str): If specified, the estimator will use this image for hosting, instead
                of selecting the appropriate official image based on framework_version.
                It can be an ECR url or dockerhub image and tag.
                Examples:
                    123.dkr.ecr.us-west-2.amazonaws.com/my-custom-image:1.0
                    custom-image:latest.
            **kwargs: Additional kwargs passed to the GluonTSModel constructor.
        Returns:
            gluonts.sagemaker.GluonTSModel: A ``GluonTSModel`` object.
            See :func:`~gluonts.sagemaker.GluonTSModel` for full details.
        """
        # TODO remove line below
        print("CREATE MODEL WAS CALLED!!!")

        return GluonTSModel(
            self.model_data,
            role or self.role,
            entry_point or self.entry_point,
            source_dir=(source_dir or self._model_source_dir()),
            enable_cloudwatch_metrics=self.enable_cloudwatch_metrics,
            name=self._current_job_name,
            container_log_level=self.container_log_level,
            code_location=self.code_location,
            framework_version=self.framework_version,
            image=(image_name or self.image_name),
            model_server_workers=model_server_workers,
            sagemaker_session=self.sagemaker_session,
            vpc_config=self.get_vpc_config(vpc_config_override),
            dependencies=(dependencies or self.dependencies),
        )

    @classmethod
    def _prepare_init_params_from_job_description(cls, job_details, model_channel_name=None):
        """Convert the job description to init params that can be handled by the
        class constructor
        Args:
            job_details: the returned job details from a describe_training_job
                API call.
            model_channel_name (str): Name of the channel where pre-trained
                model data will be downloaded.
        Returns:
            dictionary: The transformed init_params
        """
        init_params = super()._prepare_init_params_from_job_description(
            job_details, model_channel_name
        )

        """
        image_name = init_params.pop("image")
        framework, py_version, tag, _ = framework_name_from_image(image_name)

        if not framework:
            # If we were unable to parse the framework name from the image it is not one of our
            # officially supported images, in this case just add the image to the init params.
            init_params["image_name"] = image_name
            return init_params

        init_params["py_version"] = py_version

        # We switched image tagging scheme from regular image version (e.g. '1.0') to more
        # expressive containing framework version, device type and python version
        # (e.g. '0.12-gpu-py2'). For backward compatibility map deprecated image tag '1.0' to a
        # '0.12' framework version otherwise extract framework version from the tag itself.
        init_params["framework_version"] = (
            "0.12" if tag == "1.0" else framework_version_from_tag(tag)
        )

        training_job_name = init_params["base_job_name"]

        if framework != cls.__framework_name__:
            raise ValueError(
                "Training job: {} didn't use image for requested framework".format(
                    training_job_name
                )
        """
        # TODO: handle conversion from image name to params, when necessary
        return init_params

    # TODO implement
    def train(self,
              dataset: str,  # a gluonts dataset name (to be downloaded), or s3_location to a dataset in gluonts format
              # dataset
              #     ├-> train
              #     |   └> data.json
              #     ├-> test
              #     |   └> data.json
              #     └> metadata.json
              estimator: GluonEstimator,  # a gluonts estimator, gets serialized and then deserialized
              wait: bool = True,
              logs: bool = True,
              job_name: str = None
              ):
        """
        Really just a convenience wrapper for the fit(...) function: only need to specify dataset and estimator.
        The rest is taken care of. Disadvantage: somewhat limiting.
        :return:
            str: the job name
        """

        # sagemaker cant handle PosixPaths
        dataset = str(dataset)

        # pass dataset as hyper-parameter
        self._hyperparameters[self.DATASET] = dataset
        # self._hyperparameters[self.DATASET_ON_S3] = (dataset[:5] == "s3://")

        # specify job_name if not set
        if not job_name:
            milliseconds = str(int(round(time.time() * 1000)) % 1000)
            job_name = self.base_job_name + "-" + time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime()) + "-" + milliseconds

        # serialize estimator to s3
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), " Uploading - Uploading estimator config to s3.")
        s3_estimator =f"{self.code_location}/{job_name}/source/estimator.json"
        with s3fs.S3FileSystem().open(s3_estimator, 'w') as f:
            f.write(serde.dump_json(estimator))
        inputs = {"estimator": session.s3_input(s3_estimator, content_type='application/json')}

        # handle different dataset sources
        if dataset[:5] == "s3://":
            inputs.update({"s3_dataset": session.s3_input(dataset, content_type='application/json')})
        else:
            assert dataset in repository.dataset_recipes.keys(), (
                f"{dataset} is not present, please choose one from "
                f"{repository.dataset_recipes.keys()}."
            )

        # SM_CHANNEL_XXXX: A string that represents the path to the directory that contains the input
        # data for the specified channel. For example, if you specify two input channels in the MXNet

        self.fit(inputs=inputs, wait=wait, logs=logs, job_name=job_name)

        return job_name

    @classmethod
    def run(self):
        """
        If you just want to run an arbitrary script without really using the Sagemaker pipeline.
        Use cases:
            - run a model in 10 fold cross-validation
            - run multiple models
            - prototype a model
        :return:
            maybe (gluonts framework, training job arn) # might as well return the used framework
        """

    # @classmethod make this a class method later
    def tmp(self,
            inputs,  # must be valid uri apparently...
            wait: bool = True,
            logs: bool = True,
            job_name: str = None
            ):

        if not job_name:
            miliseconds = str(int(round(time.time() * 1000)) % 1000)
            job_name = self.base_job_name + "-" + time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime()) + "-" + miliseconds

        print(self.code_location)
        print(self.code_uri)

        # CAN UPLOAD DEPENDENCIES BY SPECIFYING LOCATION BEFORE FIT CALL
        self.dependencies.append('/Users/aspiele/WorkingDirectory/sagemaker_api/tmp_model')

        print("Hyperparameters:")
        print(self.hyperparameters())  # test what hyperparameters there are
        print()

        self.fit(inputs=inputs, wait=wait, logs=logs, job_name=job_name)
