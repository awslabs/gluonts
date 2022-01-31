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
    Registers the provided class with the given name in the global recommender registry.
    """

    def register(cls: R) -> R:
        RECOMMENDER_REGISTRY[name] = cls
        return cls

    return register


def register_ensemble_recommender(name: str) -> Callable[[R], R]:
    """
    Registers the provided class with the given name in the global ensemble recommender registry.
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
