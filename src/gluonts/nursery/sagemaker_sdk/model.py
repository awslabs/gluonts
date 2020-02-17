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
from typing import Dict

# Third-party imports
import sagemaker
from sagemaker.model import FrameworkModel, MODEL_SERVER_WORKERS_PARAM_NAME
from sagemaker.predictor import (
    RealTimePredictor,
    json_serializer,
    json_deserializer,
)
from sagemaker import session
from sagemaker.fw_utils import model_code_key_prefix
from pkg_resources import parse_version

# First-party imports
from .defaults import GLUONTS_VERSION, LOWEST_MMS_VERSION, FRAMEWORK_NAME


logger = logging.getLogger(__name__)


class GluonTSPredictor(RealTimePredictor):
    """
    A RealTimePredictor for inference against GluonTS Endpoints.
    This is able to serialize and deserialize datasets in the gluonts data format.
    """

    def __init__(
        self, endpoint_name: str, sagemaker_session: session.Session = None
    ):
        """Initialize an ``GluonTSPredictor``.

        Parameters
        ----------
        endpoint_name:
            The name of the endpoint to perform inference on.
        sagemaker_session :
            Session object which manages interactions with Amazon SageMaker APIs and any other
            AWS services needed. If not specified, the estimator creates one
            using the default AWS configuration chain.
        """

        # TODO: implement custom data serializer and deserializer: convert between gluonts dataset and bytes
        # Use the default functions from MXNet (they handle more than we need
        # (e.g: np.ndarrays), but that should be fine)
        super(GluonTSPredictor, self).__init__(
            endpoint_name,
            sagemaker_session,
            json_serializer,  # change this
            json_deserializer,  # change this
        )


class GluonTSModel(FrameworkModel):
    """An GluonTS SageMaker ``Model`` that can be deployed to a SageMaker ``Endpoint``."""

    __framework_name__ = FRAMEWORK_NAME
    _LOWEST_MMS_VERSION = LOWEST_MMS_VERSION

    def __init__(
        self,
        model_data,
        role,
        entry_point,
        image: str = None,
        framework_version: str = GLUONTS_VERSION,
        predictor_cls=GluonTSPredictor,  # (callable[str, session.Session])
        model_server_workers: int = None,
        **kwargs,
    ):
        """
        Initialize a GluonTSModel.

        Parameters
        ----------
        model_data:
            The S3 location of a SageMaker model data ``.tar.gz`` file.
        role:
            An AWS IAM role (either name or full ARN). The Amazon
            SageMaker training jobs and APIs that create Amazon SageMaker
            endpoints use this role to access training data and model
            artifacts. After the endpoint is created, the inference code
            might use the IAM role, if it needs to access an AWS resource.
        entry_point:
            Path (absolute or relative) to the Python source
            file which should be executed as the entry point to model
            hosting. This should be compatible with Python 3.6.
        image:
            A Docker image URI (default: None).
        framework_version:
            GluonTS version you want to use for executing your model training code.
        predictor_cls:
            A function to call to create a predictor with an endpoint name and
            SageMaker ``Session``. If specified, ``deploy()`` returns the
            result of invoking this function on the created endpoint name.
        model_server_workers:
            Optional. The number of worker processes used by the inference server. If None,
            server will use one worker per vCPU.
        **kwargs:
            Keyword arguments passed to the ``FrameworkModel`` initializer.
        """

        super(GluonTSModel, self).__init__(
            model_data,
            image,
            role,
            entry_point,
            predictor_cls=predictor_cls,
            **kwargs,
        )

        self.framework_version = framework_version
        self.model_server_workers = model_server_workers

    def prepare_container_def(
        self, instance_type, accelerator_type=None
    ) -> Dict[str, str]:
        """
        Return a container definition with framework configuration set in
        model environment variables.

        Parameters
        ----------
        instance_type:
            The EC2 instance type to deploy this Model to.
            Example::

                'ml.c5.xlarge' # CPU,
                'ml.p2.xlarge' # GPU.

        accelerator_type:
            The Elastic Inference accelerator type to
            deploy to the instance for loading and making inferences to the model.
            Example::

                "ml.eia1.medium"

        Returns
        --------
        Dict[str, str]:
            A container definition object usable with the CreateModel API.
        """

        is_mms_version = parse_version(
            self.framework_version
        ) >= parse_version(self._LOWEST_MMS_VERSION)

        deploy_image = self.image

        # TODO implement proper logic handling images when none are provided by user
        # Example implementation:
        #   https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/mxnet/model.py

        logger.info(f"Using image: {deploy_image}")

        deploy_key_prefix = model_code_key_prefix(
            self.key_prefix, self.name, deploy_image
        )
        self._upload_code(deploy_key_prefix, is_mms_version)
        deploy_env = dict(self.env)
        deploy_env.update(self._framework_env_vars())

        if self.model_server_workers:
            deploy_env[MODEL_SERVER_WORKERS_PARAM_NAME.upper()] = str(
                self.model_server_workers
            )
        return sagemaker.container_def(
            deploy_image,
            self.repacked_model_data or self.model_data,
            deploy_env,
        )
