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

from typing import Any, Callable, Dict, Type, TypeVar
from tsbench.config import EnsembleConfig, ModelConfig
from ._base import Recommender

RECOMMENDER_REGISTRY: Dict[str, Type[Recommender[ModelConfig]]] = {}
ENSEMBLE_RECOMMENDER_REGISTRY: Dict[
    str, Type[Recommender[EnsembleConfig]]
] = {}

R = TypeVar("R", bound=Type[Recommender[ModelConfig]])
E = TypeVar("E", bound=Type[Recommender[EnsembleConfig]])


def register_recommender(name: str) -> Callable[[R], R]:
    """
    Registers the provided class with the given name in the global recommender
    registry.
    """

    def register(cls: R) -> R:
        RECOMMENDER_REGISTRY[name] = cls
        return cls

    return register


def register_ensemble_recommender(name: str) -> Callable[[R], R]:
    """
    Registers the provided class with the given name in the global ensemble
    recommender registry.
    """

    def register(cls: R) -> R:
        RECOMMENDER_REGISTRY[name] = cls
        return cls

    return register


def create_recommender(name: str, **kwargs: Any) -> Recommender[ModelConfig]:
    """
    Creates a recommender using the specified parameters.
    """
    assert name in RECOMMENDER_REGISTRY, f"Unknown recommender {name}."
    recommender_cls = RECOMMENDER_REGISTRY[name]
    return recommender_cls(**kwargs)


def create_ensemble_recommender(
    name: str, **kwargs: Any
) -> Recommender[EnsembleConfig]:
    """
    Creates a recommender using the specified parameters.
    """
    assert (
        name in ENSEMBLE_RECOMMENDER_REGISTRY
    ), f"Unknown recommender {name}."
    recommender_cls = ENSEMBLE_RECOMMENDER_REGISTRY[name]
    return recommender_cls(**kwargs)
