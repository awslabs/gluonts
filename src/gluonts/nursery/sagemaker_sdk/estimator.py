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
import logging
from pathlib import Path
import time
from typing import List, Optional, Tuple, Dict
import json
import re

# Third-party imports
from sagemaker.estimator import Framework
from sagemaker.fw_utils import empty_framework_version_warning, parse_s3_url
from sagemaker.vpc_utils import VPC_CONFIG_DEFAULT
from sagemaker import session
import s3fs
import pandas as pd
import tarfile
import tempfile

# First-party imports
from gluonts.nursery.sagemaker_sdk.defaults import GLUONTS_VERSION
from gluonts.nursery.sagemaker_sdk.model import GluonTSModel
from gluonts.core import serde
from gluonts.model.estimator import GluonEstimator
from gluonts.dataset.repository import datasets
from gluonts.model.predictor import Predictor

# Defaults
ENTRY_POINTS_FOLDER = Path(__file__).parent.resolve() / "entry_point_scripts"
MONITORED_METRICS = "mean_wQuantileLoss", "ND", "RMSE"

# Logging: print log statements analogously to Sagemaker.
logger = logging.getLogger("sagemaker")


# Logging: print log statements analogously to Sagemaker.
def sagemaker_log(message):
    print(time.strftime("%Y-%m-%d %H:%M:%S ", time.gmtime()), message)


# OVERALL TODOS:
#    > TEST EVERYTHING
#    > Add python tests cases and scripts
#    > Finish documentation
#    > Add hyper parameter optimization (HPO) support
#    > Add local mode support
#    > Add officially provided images //images work now
#    > Add support for multiple instances
#    > Add support methods to easily retrieve training artifacts like the model
#    > Implement: given a hash, install that gluonts version: DONE //specify in requirements.txt
#    > Figure out dynamic install of python libraries: DONE //specify in requirements.txt


class GluonTSFramework(Framework):
    """
    This ``Estimator`` can be used to easily train and evaluate any GluonTS model on any dataset
    (own or built-in) in AWS Sagemaker using the provided Docker container.
    It also allows for the execution of custom scripts on AWS Sagemaker.
    Training is started by calling :meth:`GluonTSFramework.train` on this Estimator.
    After training is complete, calling :meth:`~sagemaker.amazon.estimator.Framework.deploy` creates a hosted
    SageMaker endpoint and returns an :class:`GluonTSPredictor` instance that can
    be used to perform inference against the hosted model.
    Alternatively, one can call the :meth:`GluonTSFramework.run` method to run a custom script defined
    by the "entry_point" argument of the :meth:`GluonTSFramework.run` method.
    Technical documentation on preparing GluonTSFramework scripts for SageMaker
    training and using the GluonTsFramework Estimator is available on the project
    home-page: https://github.com/awslabs/gluon-ts. See how_to_notebooks for examples of how to use this SDK.

    Parameters
    ----------
    sagemaker_session:
        Session object which manages interactions with Amazon SageMaker APIs
        and any other AWS services needed.
    role:
        An AWS IAM role (either name or full ARN). The Amazon SageMaker training jobs and APIs that create
        Amazon SageMaker endpoints use this role to access training data and model artifacts.
        After the endpoint is created, the inference code might use the IAM role,
        if it needs to access an AWS resource.
    image_name:
        The estimator will use this image for training and hosting. It must be an ECR url.
        If you use an image with MXNET with GPU support, you will have to
        use a GPU instance.
        Example;
        >>> '123.dkr.ecr.us-west-2.amazonaws.com/my-custom-image:1.0'
    base_job_name:
        Prefix for training job name when the :meth:`GluonTSFramework.train` or
        :meth:`GluonTSFramework.run` method is called.
    train_instance_type:
        Type of EC2 instance to use for training.
        Example;
        >>> 'ml.c5.xlarge' # CPU,
        >>> 'ml.p2.xlarge' # GPU
    train_instance_count:
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
        Example;
        >>> GluonTSFramework(entry_point='train.py', dependencies=['my/libs/common', 'requirements.txt'])
        results in the following inside the container;

        >>> opt/ml/code
        >>>     ├---> train.py
        >>>     ├---> common
        >>>     └---> requirements.txt

        To use a custom GluonTS version just import your custom GluonTS version and then call;
        >>> GluonTSFramework(entry_point='train.py', dependencies=[gluonts.__path__[0]])
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
        Not the Estimator hyperparameters, those are provided through the GluonEstimator in
        the :meth:`GluonTSFramework.train` method. If you use the :meth:`GluonTSFramework.run`
        method its up to you what you do with this parameter and you could use it to define the
        hyperparameters of your models.
        There is no support for Hyper Parameter Optimization (HPO) so far.
        In general hyperparameters will be used for training. They are made
        accessible as a dict[str, str] to the training code on
        SageMaker. For convenience, this accepts other types for keys
        and values, but ``str()`` will be called to convert them before training.
        # TODO add support for HPO
    entry_point:
        Should not be overwritten if you intend to use the :meth:`GluonTSFramework.train` method,
        and only be specified through the :meth:`GluonTSFramework.run` method.
    **kwargs:
        Additional kwargs passed to the :class:`~sagemaker.estimator.Framework` constructor.
    """

    __framework_name__ = "gluonts"
    _LOWEST_SCRIPT_MODE_VERSION = ["0", "4", "1"]

    LATEST_VERSION = "0.4.1"

    def __init__(
        self,
        sagemaker_session: session.Session,
        role: str,
        image_name: str,
        base_job_name: str,
        train_instance_type: str,
        train_instance_count: int = 1,
        dependencies: Optional[List[str]] = (),
        output_path: str = None,
        code_location: str = None,
        framework_version: str = GLUONTS_VERSION,
        hyperparameters: Dict = None,
        entry_point: str = str(ENTRY_POINTS_FOLDER / "train_entry_point.py"),
        **kwargs,
    ):
        # framework_version currently serves no purpose, except for compatibility with the sagemaker framework.
        if framework_version is None:
            logger.warning(
                empty_framework_version_warning(
                    GLUONTS_VERSION, self.LATEST_VERSION
                )
            )
        self.framework_version = framework_version or GLUONTS_VERSION

        super().__init__(
            dependencies=dependencies,
            output_path=output_path,
            code_location=code_location,
            sagemaker_session=sagemaker_session,
            role=role,
            train_instance_type=train_instance_type,
            train_instance_count=train_instance_count,
            base_job_name=base_job_name,
            entry_point=entry_point,
            hyperparameters=hyperparameters,
            image_name=image_name,
            **kwargs,
        )

        # must be set
        self.py_version = "py3"

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
        image_name: str = None,
        **kwargs,
    ) -> GluonTSModel:
        """Create a ``GluonTSModel`` object that can be deployed to an ``Endpoint``.

        Parameters
        ----------
        model_server_workers:
            The number of worker processes used by the inference server.
            If None, server will use one worker per vCPU.
        role:
            An AWS IAM role (either name or full ARN). The Amazon SageMaker training jobs and APIs that create
            Amazon SageMaker endpoints use this role to access training data and model artifacts.
            After the endpoint is created, the inference code might use the IAM role,
            if it needs to access an AWS resource.
            If not specified, the role from the Estimator will be used.
        vpc_config_override:
            Optional override for VpcConfig set on
            the model. Default: use subnets and security groups from this Estimator.
            * 'Subnets' (list[str]): List of subnet ids.
            * 'SecurityGroupIds' (list[str]): List of security group ids.
        entry_point:
            Should not be overwritten if you intend to use the :meth:`GluonTSFramework.train` method,
            and only be specified through the :meth:`GluonTSFramework.run` method.
        source_dir:
            # TODO fix later
            Path (absolute or relative) to a directory with any other serving
            source code dependencies aside from the entry point file.
            If not specified, the model source directory from training is used.
        dependencies:
            A list of paths to files or directories (absolute or relative) with any additional libraries that
            will be exported to the container. The library folders will be
            copied to SageMaker in the same folder where the "train.py" is
            copied. Include a path to a "requirements.txt" to install further dependencies at runtime.
            The provided dependencies take precedence over the pre-installed ones.
            If 'git_config' is provided, 'dependencies' should be a
            list of relative locations to directories with any additional
            libraries needed in the Git repo.
            Example;
            >>> GluonTSFramework(entry_point='train.py', dependencies=['my/libs/common', 'requirements.txt'])
            results in the following inside the container;

            >>> opt/ml/code
            >>>     ├---> train.py
            >>>     ├---> common
            >>>     └---> requirements.txt

            To use a custom GluonTS version just import your custom GluonTS version and then call;
            >>> GluonTSFramework(entry_point='train.py', dependencies=[gluonts.__path__[0]])
            This may brake the :meth:`GluonTSFramework.train` method though.
            If not specified, them dependencies from the Estimator will be used.
        image_name:
            The estimator will use this image for training and hosting. It must be an ECR url.
            If you use an image with MXNET with GPU support, you will have to
            use a GPU instance.
            Example;
            >>> '123.dkr.ecr.us-west-2.amazonaws.com/my-custom-image:1.0'
            >>> 'custom-image:latest'
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
            image=(image_name or self.image_name),
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
        # Example implementation:
        #   https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/mxnet/estimator.py

        return init_params

    @classmethod
    def _get_metrics(cls, metrics_names):
        avg_epoch_loss_metric = {
            "Name": "training_loss",
            "Regex": r"'avg_epoch_loss'=(\S+)",
        }
        final_loss_metric = {
            "Name": "final_loss",
            "Regex": r"Final loss: (\S+)",
        }
        other_metrics = [
            {
                "Name": metric,
                "Regex": rf"gluonts\[metric-{re.escape(metric)}\]: (\S+)",
            }
            for metric in metrics_names
        ]

        return [avg_epoch_loss_metric, final_loss_metric] + other_metrics

    @classmethod
    def __create_job_name(cls, base_job_name):
        milliseconds = str(int(round(time.time() * 1000)) % 1000)
        job_name = (
            base_job_name
            + "-"
            + time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())
            + "-"
            + milliseconds
        )
        return job_name

    # TODO hyperparameter override for hyper parameter optimization
    def train(
        self,
        dataset: str,
        estimator: GluonEstimator,
        num_samples: Optional[int] = 100,
        quantiles: Optional[List[int]] = (
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
        ),
        monitored_metrics: List[str] = MONITORED_METRICS,
        wait: bool = True,
        logs: bool = True,
        job_name: str = None,
    ) -> Tuple[Predictor, dict, pd.DataFrame, str]:
        """
        Use this function to train and evaluate any GluonTS model on Sagemaker. You need to call this method before
        you can call 'deploy'.

        Parameters
        ----------
        dataset:
            An s3 path-stype URL to a dataset in GluonTs format, or the name of a provided
            dataset (see gluonts.dataset.repository.datasets.dataset_recipes.keys()).
            Required dataset structure;

            >>> dataset
            >>>    ├---> train
            >>>    |   └--> data.json
            >>>    ├---> test
            >>>    |   └--> data.json
            >>>    └--> metadata.json

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

        # TODO no local mode support so far...
        if self.sagemaker_session.local_mode:
            raise NotImplementedError()

        # set metrics to be monitored
        self.metric_definitions = GluonTSFramework._get_metrics(
            monitored_metrics
        )

        # Sagemaker cant handle PosixPaths
        dataset = str(dataset)

        # pass dataset as hyper-parameter
        self._hyperparameters["DATASET"] = dataset
        self._hyperparameters["NUM_SAMPLES"] = num_samples
        self._hyperparameters["QUANTILES"] = str(quantiles)

        # specify job_name if not set
        if not job_name:
            job_name = self.__create_job_name(self.base_job_name)

        # needed to set default output and code location properly
        if self.output_path is None:
            self.output_path = (
                f"s3://{self.sagemaker_session.default_bucket()}"
            )
        sagemaker_log(f"OUTPUT_PATH: {self.output_path}/{job_name}/output")
        if self.code_location is None:
            code_bucket, _ = parse_s3_url(self.output_path)
            self.code_location = (
                f"s3://{code_bucket}"  # for consistency with sagemaker API
            )
        sagemaker_log(f"CODE_LOCATION: {self.code_location}/{job_name}/source")

        # serialize estimator to s3
        sagemaker_log("Uploading - Uploading estimator config to s3.")
        s3_estimator = f"{self.code_location}/{job_name}/source/estimator.json"
        with s3fs.S3FileSystem().open(s3_estimator, "w") as f:
            f.write(serde.dump_json(estimator))
        inputs = {
            "estimator": session.s3_input(
                s3_estimator, content_type="application/json"
            )
        }

        # handle different dataset sources
        if dataset[:5] == "s3://":
            inputs.update(
                {
                    "s3_dataset": session.s3_input(
                        dataset, content_type="application/json"
                    )
                }
            )
        else:
            assert dataset in datasets.dataset_recipes.keys(), (
                f"{dataset} is not present, please choose one from "
                f"{datasets.dataset_recipes.keys()}."
            )

        self.fit(inputs=inputs, wait=wait, logs=logs, job_name=job_name)

        # retrieve metrics
        with s3fs.S3FileSystem().open(
            f"{self.output_path}/{job_name}/output/output.tar.gz", "rb"
        ) as model_metrics:
            with tarfile.open(mode="r:gz", fileobj=model_metrics) as tar:
                agg_metrics = json.load(tar.extractfile("agg_metrics.json"))
                item_metrics = pd.read_csv(tar.extractfile("item_metrics.csv"))

        # retrieve the model itself
        temp_dir = tempfile.mkdtemp()
        with s3fs.S3FileSystem().open(
            f"{self.output_path}/{job_name}/output/model.tar.gz", "rb"
        ) as model_artifacts:
            with tarfile.open(mode="r:gz", fileobj=model_artifacts) as tar:
                tar.extractall(temp_dir)
                my_predictor = Predictor.deserialize(Path(temp_dir))

        return my_predictor, agg_metrics, item_metrics, job_name

    @classmethod
    def run(
        cls,
        entry_point: str,
        inputs,
        sagemaker_session: session.Session,
        role: str,
        image_name: str,
        base_job_name: str,
        train_instance_type: str,
        train_instance_count: int = 1,
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
            For example with the following GitHub repo directory structure;

            >>> |---> README.md
            >>> └---> src
            >>>   |---> train.py
            >>>   └---> test.py

            You can assign entry_point='src/train.py'.
        inputs:
            Type is str or dict or sagemaker.session.s3_input
            Information about the training data. This can be one of three types;

            * If (str) the S3 location where training data is saved.
            * If (dict[str, str] or dict[str, sagemaker.session.s3_input]) If using multiple
                channels for training data, you can specify a dict mapping channel names to
                strings or :func:`~sagemaker.session.s3_input` objects.
            * If (sagemaker.session.s3_input) - channel configuration for S3 data sources that can
                provide additional information as well as the path to the training dataset.
                See :func:`sagemaker.session.s3_input` for full details.
            * If (sagemaker.session.FileSystemInput) - channel configuration for
                a file system data source that can provide additional information as well as
                the path to the training dataset.

            Example:
            >>> inputs = {'my_dataset': session.s3_input(my_dataset_file, content_type='application/json')} # or
            >>> inputs = {'my_dataset': my_dataset_dir}
            where 'my_dataset_file' and 'my_dataset_dir' are the relative or absolute paths as strings.
        sagemaker_session:
            Session object which manages interactions with Amazon SageMaker APIs
            and any other AWS services needed.
        role:
            An AWS IAM role (either name or full ARN). The Amazon SageMaker training jobs and APIs that create
            Amazon SageMaker endpoints use this role to access training data and model artifacts.
            After the endpoint is created, the inference code might use the IAM role,
            if it needs to access an AWS resource.
        image_name:
            The estimator will use this image for training and hosting. It must be an ECR url.
            If you use an image with MXNET with GPU support, you will have to
            use a GPU instance.
            Example;
            >>> '123.dkr.ecr.us-west-2.amazonaws.com/my-custom-image:1.0'
        base_job_name:
            Prefix for training job name when the :meth:`GluonTSFramework.train` or
            :meth:`GluonTSFramework.run` method is called.
        train_instance_type:
            Type of EC2 instance to use for training.
            Example;
            >>> 'ml.c5.xlarge' # CPU,
            >>> 'ml.p2.xlarge' # GPU
        train_instance_count:
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
            Example;
            >>> GluonTSFramework(entry_point='train.py', dependencies=['my/libs/common', 'requirements.txt'])
            results in the following inside the container;

            >>> opt/ml/code
            >>>     ├---> train.py
            >>>     ├---> common
            >>>     └---> requirements.txt

            To use a custom GluonTS version just import your custom GluonTS version and then call;
            >>> GluonTSFramework(entry_point='train.py', dependencies=[gluonts.__path__[0]])
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
            For example with the following GitHub repo directory structure;

            >>> |---> README.md
            >>> └---> src
            >>>   |---> train.py
            >>>   └---> test.py

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
            train_instance_type=train_instance_type,
            train_instance_count=train_instance_count,
            base_job_name=base_job_name,
            image_name=image_name,
            framework_version=framework_version,
            source_dir=source_dir,
            metric_definitions=GluonTSFramework._get_metrics(
                monitored_metrics
            ),
            hyperparameters=hyperparameters,
            **kwargs,
        )

        # specify job_name if not set
        if not job_name:
            job_name = cls.__create_job_name(experiment.base_job_name)

        experiment.fit(inputs=inputs, wait=wait, logs=logs, job_name=job_name)

        return experiment, job_name
