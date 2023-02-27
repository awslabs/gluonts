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


import json
import logging
import tarfile
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

import pandas as pd
import s3fs
import sagemaker
from sagemaker.estimator import Framework
from sagemaker.s3 import parse_s3_url
from sagemaker.vpc_utils import VPC_CONFIG_DEFAULT

from gluonts.core import serde
from gluonts.dataset.repository import datasets
from gluonts.model.estimator import Estimator
from gluonts.model.predictor import Predictor

from .defaults import (
    ENTRY_POINTS_FOLDER,
    FRAMEWORK_NAME,
    GLUONTS_VERSION,
    LATEST_GLUONTS_VERSION,
    LOWEST_SCRIPT_MODE_VERSION,
    MONITORED_METRICS,
    NUM_SAMPLES,
    PYTHON_VERSION,
    QUANTILES,
    TRAIN_SCRIPT,
)
from .model import GluonTSModel
from .utils import make_job_name, make_metrics

# OVERALL TODOS:
#    > Add hyper parameter optimization (HPO) support
#    > Add local mode support
#    > Add support for multiple instances
#    > GluonTSPredictor: implement/override predict function
#    > GluonTSModel: implement correct deserialization
#    > train_entry_point.py: implement model_fn, input_fn, predict_fn, and output_fn

# HPO implementation sketch:
#    > Example HPO of model: MODEL_HPM:Trainer:batch_size:64
#    > Now construct nested dict from MODEL_HPM hyperparameters
#    > Load the serialized model as a dict
#    > Update the model dict with the nested dict from the MODEL_HPMs
#      with dict.update(...)
#    > Write this new dict back to a s3 as a .json file like before


logger = logging.getLogger(__name__)


class TrainResult(NamedTuple):
    predictor: Predictor
    metrics: tuple
    job_name: str


class Locations(NamedTuple):
    job_name: str
    output_path: str
    code_location: str

    @property
    def job_output_path(self):
        return f"{self.output_path}/{self.job_name}/output"

    @property
    def job_code_location(self):
        return f"{self.code_location}/{self.job_name}/source"

    @property
    def estimator_path(self):
        return f"{self.job_code_location}/estimator.json"

    @property
    def output_archive(self):
        return f"{self.job_output_path}/output.tar.gz"

    @property
    def model_archive(self):
        return f"{self.job_output_path}/model.tar.gz"


class GluonTSFramework(Framework):
    """
    This ``Estimator`` can be used to easily train and evaluate any GluonTS
    model on any dataset (own or built-in) in AWS Sagemaker using the provided
    Docker container. It also allows for the execution of custom scripts on AWS
    Sagemaker. Training is started by calling :meth:`GluonTSFramework.train` on
    this Estimator. After training is complete, calling
    :meth:`~sagemaker.amazon.estimator.Framework.deploy` creates a hosted
    SageMaker endpoint and returns an :class:`GluonTSPredictor` instance that
    can be used to perform inference against the hosted model. Alternatively,
    one can call the :meth:`GluonTSFramework.run` method to run a custom script
    defined by the "entry_point" argument of the :meth:`GluonTSFramework.run`
    method. Technical documentation on preparing GluonTSFramework scripts for
    SageMaker training and using the GluonTsFramework Estimator is available on
    the project home-page: https://github.com/awslabs/gluon-ts. See
    how_to_notebooks for examples of how to use this SDK.

    Parameters
    ----------
    sagemaker_session:
        Session object which manages interactions with Amazon SageMaker APIs
        and any other AWS services needed.
    role:
        An AWS IAM role (either name or full ARN). The Amazon SageMaker
        training jobs and APIs that create Amazon SageMaker endpoints use this
        role to access training data and model artifacts. After the endpoint is
        created, the inference code might use the IAM role, if it needs to
        access an AWS resource.
    image_uri:
        The estimator will use this image for training and hosting. It must be
        an ECR url. If you use an image with MXNET with GPU support, you will
        have to use a GPU instance.
        Example::

            '123.dkr.ecr.us-west-2.amazonaws.com/my-custom-image:1.0'

    base_job_name:
        Prefix for training job name when the :meth:`GluonTSFramework.train` or
        :meth:`GluonTSFramework.run` method is called.
    instance_type:
        Type of EC2 instance to use for training.
        Example::

            'ml.c5.xlarge' # CPU,
            'ml.p2.xlarge' # GPU

    instance_count:
        Currently not more than one supported.
        Otherwise the number of Amazon EC2 instances to use for training.
    dependencies:
        A list of paths to files or directories (absolute or relative) with any
        additional libraries that will be exported to the container. The
        library folders will be copied to SageMaker in the same folder where
        the "train.py" is copied. Include a path to a "requirements.txt" to
        install further dependencies at runtime. The provided dependencies take
        precedence over the pre-installed ones. If 'git_config' is provided,
        'dependencies' should be a list of relative locations to directories
        with any additional libraries needed in the Git repo.
        Example::

            GluonTSFramework(
                entry_point='train.py',
                dependencies=['my/libs/common', 'requirements.txt']
            )

        results in the following inside the container::

            opt/ml/code
                 ├---> train.py
                 ├---> common
                 └---> requirements.txt

        To use a custom GluonTS version just import your custom GluonTS version
        and then call::

             GluonTSFramework(
                entry_point='train.py',
                dependencies=[gluonts.__path__[0]]
            )

        This may brake the :meth:`GluonTSFramework.train` method though. If not
        specified, them dependencies from the Estimator will be used.
    output_path:
        S3 location for saving the transform result. If not specified, results
        are stored to a default bucket.
    code_location:
        The S3 prefix URI where custom code will be uploaded. The code file
        uploaded in S3 is 'code_location/source/sourcedir.tar.gz'. If not
        specified, the default code location is s3://default_bucket/job-name/.
        And code file uploaded to S3 is
        s3://default_bucket/job-name/source/sourcedir.tar.gz
    framework_version:
        GluonTS version. If not specified, this will default to 0.4.1.
        Currently has no effect.
    hyperparameters:
        # TODO add support for HPO
        Not the Estimator hyperparameters, those are provided through the
        Estimator in the :meth:`GluonTSFramework.train` method. If you use the
        :meth:`GluonTSFramework.run` method its up to you what you do with this
        parameter and you could use it to define the hyperparameters of your
        models. There is no support for Hyper Parameter Optimization (HPO) so
        far. In general hyperparameters will be used for training. They are
        made accessible as a dict[str, str] to the training code on SageMaker.
        For convenience, this accepts other types for keys and values, but
        ``str()`` will be called to convert them before training.
    entry_point:
        Should not be overwritten if you intend to use the
        :meth:`GluonTSFramework.train` method, and only be specified through
        the :meth:`GluonTSFramework.run` method.
    **kwargs:
        Additional kwargs passed to the :class:`~sagemaker.estimator.Framework`
        constructor.
    """

    __framework_name__ = FRAMEWORK_NAME
    _LOWEST_SCRIPT_MODE_VERSION = LOWEST_SCRIPT_MODE_VERSION
    LATEST_VERSION = LATEST_GLUONTS_VERSION

    def __init__(
        self,
        sagemaker_session: sagemaker.Session,
        role: str,
        image_uri: str,
        base_job_name: str,
        instance_type: str = "ml.c5.xlarge",
        instance_count: int = 1,
        dependencies: Optional[List[str]] = None,
        output_path: str = None,
        code_location: str = None,
        framework_version: str = GLUONTS_VERSION,
        hyperparameters: Dict = None,
        entry_point: str = str(ENTRY_POINTS_FOLDER / TRAIN_SCRIPT),
        **kwargs,
    ):
        # Framework_version currently serves no purpose,
        # except for compatibility with the sagemaker framework.
        self.framework_version = framework_version or GLUONTS_VERSION

        super().__init__(
            dependencies=dependencies,
            output_path=output_path,
            code_location=code_location,
            sagemaker_session=sagemaker_session,
            role=role,
            instance_type=instance_type,
            instance_count=instance_count,
            base_job_name=base_job_name,
            entry_point=entry_point,
            hyperparameters=hyperparameters,
            image_uri=image_uri,
            **kwargs,
        )

        # must be set
        self.py_version = PYTHON_VERSION

        self._s3fs = s3fs.S3FileSystem(
            session=sagemaker_session.boto_session._session
        )

    def create_model(
        self,
        model_server_workers: Optional[str] = None,
        role: str = None,
        vpc_config_override: Optional[
            Dict[str, List[str]]
        ] = VPC_CONFIG_DEFAULT,
        entry_point: str = None,
        source_dir: str = None,
        dependencies: List[str] = None,
        image_uri: str = None,
        **kwargs,
    ) -> GluonTSModel:
        """Create a ``GluonTSModel`` object that can be deployed to an
        ``Endpoint``.

        Parameters
        ----------
        model_server_workers:
            The number of worker processes used by the inference server. If
            None, server will use one worker per vCPU.
        role:
            An AWS IAM role (either name or full ARN). The Amazon SageMaker
            training jobs and APIs that create Amazon SageMaker endpoints use
            this role to access training data and model artifacts. After the
            endpoint is created, the inference code might use the IAM role, if
            it needs to access an AWS resource. If not specified, the role from
            the Estimator will be used.
        vpc_config_override:
            Optional override for VpcConfig set on the model. Default: use
            subnets and security groups from this Estimator.
            * 'Subnets' (list[str]): List of subnet ids.
            * 'SecurityGroupIds' (list[str]): List of security group ids.
        entry_point:
            Should not be overwritten if you intend to use the
            :meth:`GluonTSFramework.train` method, and only be specified
            through the :meth:`GluonTSFramework.run` method.
        source_dir:
            If you set this, your training script will have to be located
            within the specified source_dir and you will have to set
            entry_point to the relative path within your source_dir.

            Path (absolute, relative, or an S3 URI) to a directory with all
            training source code including dependencies. Structure within this
            directory is preserved when training on Amazon SageMaker. If
            'git_config' is provided, 'source_dir' should be a relative
            location to a directory in the Git repo. For example with the
            following GitHub repo directory structure::

                |---> README.md
                └---> src
                  |---> train.py
                  └---> test.py

            and you need 'train.py' as entry point and 'test.py' as training
            source code as well, you must set entry_point='train.py',
            source_dir='src'. If not specified, the model source directory from
            training is used.
        dependencies:
            A list of paths to files or directories (absolute or relative) with
            any additional libraries that will be exported to the container.
            The library folders will be copied to SageMaker in the same folder
            where the "train.py" is copied. Include a path to a
            "requirements.txt" to install further dependencies at runtime. The
            provided dependencies take precedence over the pre-installed ones.
            If 'git_config' is provided, 'dependencies' should be a list of
            relative locations to directories with any additional libraries
            needed in the Git repo.
            Example::

                GluonTSFramework(
                    entry_point='train.py',
                    dependencies=['my/libs/common', 'requirements.txt']
                )

            results in the following inside the container::

                opt/ml/code
                    ├---> train.py
                    ├---> common
                    └---> requirements.txt

            To use a custom GluonTS version just import your custom GluonTS
            version and then call::

                 GluonTSFramework(
                    entry_point='train.py',
                    dependencies=[gluonts.__path__[0]]
                )

            This may brake the :meth:`GluonTSFramework.train` method though.
            If not specified, them dependencies from the Estimator will be
            used.
        image_uri:
            The estimator will use this image for training and hosting. It must
            be an ECR url. If you use an image with MXNET with GPU support, you
            will have to use a GPU instance.
            Example::

                 '123.dkr.ecr.us-west-2.amazonaws.com/my-custom-image:1.0'
                 'custom-image:latest'

            If not specified, them image from the Estimator will be used.
        **kwargs:
            Additional kwargs passed to the GluonTSModel constructor.

        Returns
        -------
            gluonts.sagemaker.GluonTSModel
                A ``GluonTSModel`` object.
                See :func:`~gluonts.sagemaker.GluonTSModel` for full details.
        """

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
            image_uri=(image_uri or self.image_uri),
            model_server_workers=model_server_workers,
            sagemaker_session=self.sagemaker_session,
            vpc_config=self.get_vpc_config(vpc_config_override),
            dependencies=(dependencies or self.dependencies),
        )

    @classmethod
    def _prepare_init_params_from_job_description(
        cls, job_details, model_channel_name: str = None
    ):
        """
        Convert the job description to init params that can be handled by the
        class constructor

        Parameters
        ----------
        job_details:
            the returned job details from a describe_training_job API call.
        model_channel_name:
            Name of the channel where pre-trained model data will be downloaded.

        Returns
        -------
        Dict:
            The transformed init_params
        """

        init_params = super()._prepare_init_params_from_job_description(
            job_details, model_channel_name
        )

        # TODO: handle conversion from image name to params, once default
        # images are provided
        # Example implementation:
        #   https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/mxnet/estimator.py

        return init_params

    def _initialize_job(
        self, monitored_metrics, dataset, num_samples, quantiles, job_name
    ):
        if self.sagemaker_session.local_mode:
            # TODO implement local mode support
            raise NotImplementedError(
                "Local mode has not yet been implemented."
            )

        # set metrics to be monitored
        self.metric_definitions = make_metrics(monitored_metrics)

        self._hyperparameters.update(
            DATASET=dataset,  # pass dataset as hyper-parameter
            NUM_SAMPLES=num_samples,
            QUANTILES=str(quantiles),
        )

        # needed to set default output and code location properly
        if self.output_path is None:
            default_bucket = self.sagemaker_session.default_bucket()
            self.output_path = f"s3://{default_bucket}"

        if self.code_location is None:
            code_bucket, _ = parse_s3_url(self.output_path)
            self.code_location = (
                f"s3://{code_bucket}"  # for consistency with sagemaker API
            )

        locations = Locations(
            job_name=job_name,
            output_path=self.output_path,
            code_location=self.code_location,
        )

        logger.info(f"OUTPUT_PATH: {locations.job_output_path}")
        logger.info(f"CODE_LOCATION: {locations.job_code_location}")

        return locations

    def _upload_estimator(self, locations, estimator):
        logger.info("Uploading estimator config to s3.")

        serialized = serde.dump_json(estimator)

        with self._s3fs.open(locations.estimator_path, "w") as estimator_file:
            estimator_file.write(serialized)

    def _prepare_inputs(self, locations, dataset):
        s3_json_input = partial(
            sagemaker.TrainingInput, content_type="application/json"
        )

        inputs = {"estimator": s3_json_input(locations.estimator_path)}

        if dataset.startswith("s3://"):
            inputs["s3_dataset"] = s3_json_input(dataset)
        else:
            assert dataset in datasets.dataset_recipes, (
                f"{dataset} is not present, please choose one from "
                f"{list(datasets.dataset_recipes)}."
            )

        return inputs

    def _retrieve_metrics(self, locations):
        with self._s3fs.open(locations.output_archive, "rb") as stream:
            with tarfile.open(fileobj=stream, mode="r:gz") as archive:
                agg_metrics = json.load(
                    archive.extractfile("agg_metrics.json")
                )
                item_metrics = pd.read_csv(
                    archive.extractfile("item_metrics.csv")
                )

        return agg_metrics, item_metrics

    def _retrieve_model(self, locations):
        with self._s3fs.open(locations.model_archive, "rb") as stream:
            with tarfile.open(mode="r:gz", fileobj=stream) as archive:
                with TemporaryDirectory() as temp_dir:
                    archive.extractall(temp_dir)
                    predictor = Predictor.deserialize(Path(temp_dir))

        return predictor

    # TODO hyperparameter override for hyper parameter optimization
    def train(
        self,
        dataset: str,
        estimator: Estimator,
        num_samples: int = NUM_SAMPLES,
        quantiles: List[float] = QUANTILES,
        monitored_metrics: List[str] = MONITORED_METRICS,
        wait: bool = True,
        logs: bool = True,
        job_name: str = None,
    ) -> Union[TrainResult, str]:
        """
        Use this function to train and evaluate any GluonTS model on Sagemaker.
        You need to call this method before you can call 'deploy'.

        Parameters
        ----------
        dataset:
            An s3 path-stype URL to a dataset in GluonTs format, or the name of
            a provided dataset (see
            gluonts.dataset.repository.datasets.dataset_recipes.keys()).
            Required dataset structure::

                dataset
                    ├---> train
                    |   └--> data.json
                    ├---> test
                    |   └--> data.json
                    └--> metadata.json

        estimator:
            The GluonTS estimator that should be trained. If you want to train a custom estimator
            you must have specified the code location in the dependencies argument of the GLuonTSFramework.
        num_samples:
            The num_samples parameter for the gluonts.evaluation.backtest.make_evaluation_predictions
            method used for evaluation. (default: (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9))
        quantiles:
            The quantiles parameter for the gluonts.evaluation.Evaluator used
            for evaluation. (default: 0.1)
        monitored_metrics:
            Names of the metrics that will be parsed from logs in a one minute interval
            in order to monitor them in Sagemaker.
        wait:
            Whether the call should wait until the job completes (default: True).
        logs:
            Whether to show the logs produced by the job. Only meaningful when wait is True (default: True).
        job_name:
            Training job name. If not specified, a default job name will be generated,
            based on the base_job_name and the current timestamp.

        Returns
        --------
        job_name
            The job name used during training.
        """

        if not job_name:
            job_name = make_job_name(self.base_job_name)

        locations = self._initialize_job(
            monitored_metrics, dataset, num_samples, quantiles, job_name
        )
        self._upload_estimator(locations, estimator)

        inputs = self._prepare_inputs(locations, dataset)
        self.fit(inputs=inputs, wait=wait, logs=logs, job_name=job_name)

        if wait:
            metrics = self._retrieve_metrics(locations)
            predictor = self._retrieve_model(locations)

            return TrainResult(
                predictor=predictor, metrics=metrics, job_name=job_name
            )
        else:
            return job_name

    @classmethod
    def run(
        cls,
        entry_point: str,
        inputs,
        sagemaker_session: sagemaker.Session,
        role: str,
        image_uri: str,
        base_job_name: str,
        instance_type: str,
        instance_count: int = 1,
        dependencies: Optional[List[str]] = [],
        output_path: str = None,
        code_location: str = None,
        framework_version: str = GLUONTS_VERSION,
        hyperparameters=None,
        source_dir: str = None,
        monitored_metrics: List[str] = MONITORED_METRICS,
        wait: bool = True,
        logs: bool = True,
        job_name: str = None,
        **kwargs,
    ) -> Tuple[Framework, str]:
        """
        Use this function to run a custom script specified in 'entry_point' in GluonTSFramework.
        To access files on s3 specify them in inputs. If you want to access local files you should
        have specified them in 'dependencies' in GluonTSFramework.

        Parameters
        ----------
        entry_point:
            Path (absolute or relative) to the local Python source file which should be executed as the entry point to
            training. This should be compatible with Python 3.6. If 'git_config' is provided, 'entry_point' should be
            a relative location to the Python source file in the Git repo.
            For example with the following GitHub repo directory structure::

                |---> README.md
                └---> src
                    |---> train.py
                    └---> test.py

            You can assign entry_point='src/train.py'.
        inputs:
            Type is str or dict or sagemaker.TrainingInput, however, cannot be empty!
            Information about the training data. This can be one of three types;

            * If (str) the S3 location where training data is saved.
            * If (dict[str, str] or dict[str, sagemaker.TrainingInput]) If using multiple
                channels for training data, you can specify a dict mapping channel names to
                strings or :func:`~sagemaker.TrainingInput` objects.
            * If (sagemaker.TrainingInput) - channel configuration for S3 data sources that can
                provide additional information as well as the path to the training dataset.
                See :func:`sagemaker.TrainingInput` for full details.
            * If (sagemaker.session.FileSystemInput) - channel configuration for
                a file system data source that can provide additional information as well as
                the path to the training dataset.

            Example::

                inputs = {'my_dataset': sagemaker.TrainingInput(my_dataset_file, content_type='application/json')} # or
                inputs = {'my_dataset': my_dataset_dir}

            where 'my_dataset_file' and 'my_dataset_dir' are the relative or absolute paths as strings.
        sagemaker_session:
            Session object which manages interactions with Amazon SageMaker APIs
            and any other AWS services needed.
        role:
            An AWS IAM role (either name or full ARN). The Amazon SageMaker training jobs and APIs that create
            Amazon SageMaker endpoints use this role to access training data and model artifacts.
            After the endpoint is created, the inference code might use the IAM role,
            if it needs to access an AWS resource.
        image_uri:
            The estimator will use this image for training and hosting. It must be an ECR url.
            If you use an image with MXNET with GPU support, you will have to
            use a GPU instance.
            Example::

                '123.dkr.ecr.us-west-2.amazonaws.com/my-custom-image:1.0'

        base_job_name:
            Prefix for training job name when the :meth:`GluonTSFramework.train` or
            :meth:`GluonTSFramework.run` method is called.
        instance_type:
            Type of EC2 instance to use for training.
            Example::

                'ml.c5.xlarge' # CPU,
                'ml.p2.xlarge' # GPU

        instance_count:
            Currently not more than one supported.
            Otherwise the number of Amazon EC2 instances to use for training.
        dependencies:
            A list of paths to files or directories (absolute or relative) with any additional libraries that
            will be exported to the container. The library folders will be
            copied to SageMaker in the same folder where the "train.py" is
            copied. Include a path to a "requirements.txt" to install further dependencies at runtime.
            The provided dependencies take precedence over the pre-installed ones.
            If 'git_config' is provided, 'dependencies' should be a
            list of relative locations to directories with any additional
            libraries needed in the Git repo.
            Example::

                GluonTSFramework.run(entry_point='train.py', dependencies=['my/libs/common', 'requirements.txt'])

            results in the following inside the container::

                opt/ml/code
                     ├---> train.py
                     ├---> common
                     └---> requirements.txt

            To use a custom GluonTS version just import your custom GluonTS version and then call::

                 GluonTSFramework.run(entry_point='train.py', dependencies=[gluonts.__path__[0]])

            This may brake the :meth:`GluonTSFramework.train` method though.
            If not specified, them dependencies from the Estimator will be used.
        output_path:
            S3 location for saving the transform result. If not specified,
            results are stored to a default bucket.
        code_location:
            The S3 prefix URI where custom code will be
            uploaded. The code file uploaded in S3 is 'code_location/source/sourcedir.tar.gz'.
            If not specified, the default code location is s3://default_bucket/job-name/.
            And code file uploaded to S3 is s3://default_bucket/job-name/source/sourcedir.tar.gz
        framework_version:
            GluonTS version. If not specified, this will default to 0.4.1. Currently has no effect.
        hyperparameters:
            Its up to you what you do with this parameter and you could use it to define the
            hyperparameters of your models.
            In general hyperparameters will be used for training. They are made
            accessible as a dict[str, str] to the training code on
            SageMaker. For convenience, this accepts other types for keys
            and values, but ``str()`` will be called to convert them before training.
        source_dir:
            If you set this, your training script will have to be located within the
            specified source_dir and you will have to set entry_point to the relative path within
            your source_dir.
            Path (absolute, relative, or an S3 URI) to a directory with all training source code
            including dependencies. Structure within this directory is preserved when training on
            Amazon SageMaker. If 'git_config' is provided, 'source_dir' should be a relative
            location to a directory in the Git repo.
            For example with the following GitHub repo directory structure::

                |---> README.md
                └---> src
                  |---> train.py
                  └---> test.py

            and you need 'train.py' as entry point and 'test.py' as training source code as well,
            you must set entry_point='train.py', source_dir='src'.
        monitored_metrics:
            Names of the metrics that will be parsed from logs in a one minute interval
            in order to monitor them in Sagemaker.
        wait:
            Whether the call should wait until the job completes (default: True).
        logs:
            Whether to show the logs produced by the job. Only meaningful when wait is True (default: True).
        job_name:
            Training job name. If not specified, a default job name will be generated,
            based on the base_job_name and the current timestamp.
        Returns
        --------
            Tuple[Framework, str]:
                The GluonTSFramework and the job name of the training job.
        """

        experiment = GluonTSFramework(
            entry_point=entry_point,
            dependencies=dependencies,
            output_path=output_path,
            code_location=code_location,
            sagemaker_session=sagemaker_session,
            role=role,
            instance_type=instance_type,
            instance_count=instance_count,
            base_job_name=base_job_name,
            image_uri=image_uri,
            framework_version=framework_version,
            source_dir=source_dir,
            metric_definitions=make_metrics(monitored_metrics),
            hyperparameters=hyperparameters,
            **kwargs,
        )

        if not job_name:
            job_name = make_job_name(experiment.base_job_name)

        experiment.fit(inputs=inputs, wait=wait, logs=logs, job_name=job_name)

        return experiment, job_name
