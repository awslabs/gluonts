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

# Standard library imports

# Third-party imports
from sagemaker.model import FrameworkModel
from sagemaker.predictor import RealTimePredictor, json_serializer, json_deserializer
# import sagemaker
# from sagemaker.fw_utils import create_image_uri, model_code_key_prefix, python_deprecation_warning
# MODEL_SERVER_WORKERS_PARAM_NAME

# First-party imports
from gluonts.sagemaker.defaults import GLUONTS_VERSION


class GluonTSPredictor(RealTimePredictor):
    """A RealTimePredictor for inference against GluonTS Endpoints.
    This is able to serialize and deserialize datasets in the gluonts data format.
    """

    def __init__(self, endpoint_name, sagemaker_session=None):
        """Initialize an ``GluonTSPredictor``.
        Args:
            endpoint_name (str): The name of the endpoint to perform inference
                on.
            sagemaker_session (sagemaker.session.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, the estimator creates one
                using the default AWS configuration chain.
        """

        # Use the default functions from MXNet (they handle more than we need
        # (e.g: np.ndarrays), but that should be fine)
        super(GluonTSPredictor, self).__init__(
            endpoint_name, sagemaker_session, json_serializer, json_deserializer
        )


class GluonTSModel(FrameworkModel):
    """An GluonTS SageMaker ``Model`` that can be deployed to a SageMaker ``Endpoint``."""

    __framework_name__ = "gluonts"
    _LOWEST_MMS_VERSION = "0.3.3" # TODO: figure out the meaning of this...

    def __init__(
        self,
        model_data,
        role,
        entry_point,
        image=None,
        framework_version=GLUONTS_VERSION,
        predictor_cls=GluonTSPredictor,
        model_server_workers=None,
        **kwargs
    ):
        """Initialize an MXNetModel.
        Args:
            model_data (str): The S3 location of a SageMaker model data
                ``.tar.gz`` file.
            role (str): An AWS IAM role (either name or full ARN). The Amazon
                SageMaker training jobs and APIs that create Amazon SageMaker
                endpoints use this role to access training data and model
                artifacts. After the endpoint is created, the inference code
                might use the IAM role, if it needs to access an AWS resource.
            entry_point (str): Path (absolute or relative) to the Python source
                file which should be executed as the entry point to model
                hosting. This should be compatible with Python 3.6.
            image (str): A Docker image URI (default: None). If not specified, a
                default image for GluonTSFramework will be used.
            framework_version (str): GluonTS version you want to use for executing
                your model training code.
            predictor_cls (callable[str, sagemaker.session.Session]): A function
                to call to create a predictor with an endpoint name and
                SageMaker ``Session``. If specified, ``deploy()`` returns the
                result of invoking this function on the created endpoint name.
            model_server_workers (int): Optional. The number of worker processes
                used by the inference server. If None, server will use one
                worker per vCPU.
            **kwargs: Keyword arguments passed to the ``FrameworkModel``
                initializer.
        """
        super(GluonTSModel, self).__init__(
            model_data, image, role, entry_point, predictor_cls=predictor_cls, **kwargs
        )

        self.framework_version = framework_version
        self.model_server_workers = model_server_workers

    def prepare_container_def(self, instance_type, accelerator_type=None):
        """Return a container definition with framework configuration set in
        model environment variables.
        Args:
            instance_type (str): The EC2 instance type to deploy this Model to.
                For example, 'ml.p2.xlarge'.
            accelerator_type (str): The Elastic Inference accelerator type to
                deploy to the instance for loading and making inferences to the
                model. For example, 'ml.eia1.medium'.
        Returns:
            dict[str, str]: A container definition object usable with the
            CreateModel API.
        """

        # Code from MXNet implementation:
        """
        is_mms_version = parse_version(self.framework_version) >= parse_version(
            self._LOWEST_MMS_VERSION
        )

        deploy_image = self.image
        if not deploy_image:
            region_name = self.sagemaker_session.boto_session.region_name

            framework_name = self.__framework_name__
            if is_mms_version:
                framework_name += "-serving"

            deploy_image = create_image_uri(
                region_name,
                framework_name,
                instance_type,
                self.framework_version,
                accelerator_type=accelerator_type,
            )

        deploy_key_prefix = model_code_key_prefix(self.key_prefix, self.name, deploy_image)
        self._upload_code(deploy_key_prefix, is_mms_version)
        deploy_env = dict(self.env)
        deploy_env.update(self._framework_env_vars())

        if self.model_server_workers:
            deploy_env[MODEL_SERVER_WORKERS_PARAM_NAME.upper()] = str(self.model_server_workers)
        return sagemaker.container_def(
            deploy_image, self.repacked_model_data or self.model_data, deploy_env
        )
        """

        # TODO implement the proper logic for handling images
        # Example implementation:
        #   https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/mxnet/model.py

        return super().prepare_container_def(instance_type, accelerator_type)
