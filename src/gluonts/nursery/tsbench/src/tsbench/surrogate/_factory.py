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

from typing import Any, Callable, Dict, List, Optional, Type, TypeVar
from tsbench.config import EnsembleConfig, ModelConfig
from tsbench.evaluations.tracking import EnsembleTracker, ModelTracker
from ._base import DatasetFeaturesMixin, Surrogate

SURROGATE_REGISTRY: Dict[str, Type[Surrogate[ModelConfig]]] = {}
ENSEMBLE_SURROGATE_REGISTRY: Dict[str, Type[Surrogate[EnsembleConfig]]] = {}


S = TypeVar("S", bound=Type[Surrogate[ModelConfig]])
E = TypeVar("E", bound=Type[Surrogate[EnsembleConfig]])


def register_surrogate(name: str) -> Callable[[S], S]:
    """
    Registers the provided class with the given name in the global surrogate
    registry.
    """

    def register(cls: S) -> S:
        SURROGATE_REGISTRY[name] = cls
        return cls

    return register


def register_ensemble_surrogate(name: str) -> Callable[[E], E]:
    """
    Registers the provided class with the given name in the global ensemble
    surrogate registry.
    """

    def register(cls: E) -> E:
        ENSEMBLE_SURROGATE_REGISTRY[name] = cls
        return cls

    return register


def create_surrogate(
    name: str,
    tracker: ModelTracker,
    predict: Optional[List[str]],
    input_flags: Dict[str, bool],
    **kwargs: Any,
) -> Surrogate[ModelConfig]:
    """
    Creates a surrogate using the specified parameters.
    """
    assert name in SURROGATE_REGISTRY, f"Unknown surrogate {name}."

    # Build the parameters
    surrogate_cls = SURROGATE_REGISTRY[name]
    args = {"predict": predict, "tracker": tracker, **kwargs}
    if issubclass(surrogate_cls, DatasetFeaturesMixin):
        args.update(input_flags)

    # Initialize the surrogate
    return surrogate_cls(**args)  # type: ignore


def create_ensemble_surrogate(
    name: str,
    tracker: EnsembleTracker,
    predict: Optional[List[str]],
    input_flags: Dict[str, bool],
    **kwargs: Any,
) -> Surrogate[EnsembleConfig]:
    """
    Creates an ensemble surrogate using the specified parameters.
    """
    assert name in ENSEMBLE_SURROGATE_REGISTRY, f"Unknown surrogate {name}."

    # Build the parameters
    surrogate_cls = ENSEMBLE_SURROGATE_REGISTRY[name]
    args = {"predict": predict, "tracker": tracker, **kwargs}
    if issubclass(surrogate_cls, DatasetFeaturesMixin):
        args.update(input_flags)

    # Initialize the surrogate
    return surrogate_cls(**args)  # type: ignore
