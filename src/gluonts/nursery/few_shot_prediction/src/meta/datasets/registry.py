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
import pytorch_lightning as pl
from gluonts.dataset.repository.datasets import dataset_recipes


DATA_MODULE_REGISTRY: Dict[str, Type[pl.LightningDataModule]] = {}


M = TypeVar("M", bound=Type[pl.LightningDataModule])


def register_data_module(cls: M) -> M:
    """
    Registers the provided class in the global model registry.
    """
    DATA_MODULE_REGISTRY[cls.name()] = cls
    return cls


def get_data_module(name: str, **kwargs: Any) -> pl.LightningDataModule:
    """
    This method creates the data module with the specified name. The provided
    keyword arguments must contain ALL arguments required by the model configuration.
    Superfluous arguments may be provided and are simply ignored.

    Parameters
    ----------
    name: str
        The canonical name of the data module. See `DATA_MODULE_REGISTRY`.
    kwargs: Any
        Keyword arguments passed to the initializer of the data module.

    Returns
    -------
    ModelConfig
        The data module.
    """
    if name in DATA_MODULE_REGISTRY:
        dm_module_cls = DATA_MODULE_REGISTRY[name]
    elif name.startswith("dm_m4"):
        dm_module_cls = DATA_MODULE_REGISTRY["dm_m4"]
    elif name.startswith("dm_m3"):
        dm_module_cls = DATA_MODULE_REGISTRY["dm_m3"]
    elif name.startswith("dm_m1"):
        dm_module_cls = DATA_MODULE_REGISTRY["dm_m1"]
    elif name[3:] in dataset_recipes.keys():
        dm_module_cls = DATA_MODULE_REGISTRY["dm_gluonts"]
    else:
        print(f"Data module name {name} is unknown.")

    return dm_module_cls(**kwargs)
    # return dm_module_cls(
    #     **{k: v for k, v in kwargs.items() if k in dm_module_cls.__init__.__annotations__}
    # )
