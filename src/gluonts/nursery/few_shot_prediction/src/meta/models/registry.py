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

from typing import Dict, Type, TypeVar, Any
import torch.nn as nn
from lightkit.utils import get_generic_type
from lightkit.nn import Configurable
from dataclasses import MISSING

MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {}


M = TypeVar("M", bound=Type[nn.Module])


def register_model(cls: M) -> M:
    """
    Registers the provided class in the global model registry.
    """
    MODEL_REGISTRY[cls.name()] = cls
    return cls


def get_model(name: str, **kwargs: Any) -> nn.Module:
    """
    This method creates the model configuration of the model with the specified name. The provided
    keyword arguments must contain ALL arguments required by the model configuration. Superfluous
    arguments may be provided and are simply ignored.

    In case the name is unknown or parameters for the model config's initializer are missing, an
    assertion error occurs.

    Parameters
    ----------
    name: str
        The canonical name of the model. See `MODEL_REGISTRY`.
    kwargs: Any
        Keyword arguments passed to the initializer of the model config.

    Returns
    -------
    ModelConfig
        The model configuration.
    """
    # Get the model
    assert name in MODEL_REGISTRY, f"Model name '{name}' is unknown."
    model_cls = MODEL_REGISTRY[name]
    config_cls = get_generic_type(model_cls, Configurable)

    # Filter the required parameters
    all_params = set(key for key in config_cls.__dataclass_fields__.keys())
    required = {
        key
        for key, field in config_cls.__dataclass_fields__.items()
        if (field.default is MISSING and field.default_factory is MISSING)  # type: ignore
    }
    assert all(
        r in kwargs for r in required
    ), "Keyword arguments missing at least one parameter."

    return model_cls(
        config_cls(**{k: v for k, v in kwargs.items() if k in all_params})
    )
