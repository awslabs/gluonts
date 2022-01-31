from typing import Any
from sagemaker.estimator import Framework
from sagemaker.model import Model


class CustomFramework(Framework):  # type: ignore
    """
    A custom framework is a dummy implementation which allows instantiating a custom AWS Sagemaker
    framework.
    """

    _framework_name = "custom"

    def create_model(self, **kwargs: Any) -> Model:
        raise NotImplementedError
