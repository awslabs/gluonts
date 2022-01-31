from typing import Any, Dict, Type, TypeVar
from ._base import ModelConfig

MODEL_REGISTRY: Dict[str, Type[ModelConfig]] = {}


M = TypeVar("M", bound=Type[ModelConfig])


def register_model(cls: M) -> M:
    """
    Registers the provided class in the global model registry.
    """
    MODEL_REGISTRY[cls.name()] = cls
    return cls


def get_model_config(name: str, **kwargs: Any) -> ModelConfig:
    """
    This method creates the model configuration of the model with the specified name. The provided
    keyword arguments must contain ALL arguments required by the model configuration. Superfluous
    arguments may be provided and are simply ignored.

    In case the name is unknown or parameters for the model config's initializer are missing, an
    assertion error occurs.

    Args:
        name: The canonical name of the model. See `MODEL_REGISTRY`.
        kwargs: Keyword arguments passed to the initializer of the model config.

    Returns:
        The model configuration.
    """
    # Get the model
    assert name in MODEL_REGISTRY, f"Model name '{name}' is unknown."
    config_cls = MODEL_REGISTRY[name]

    # Filter the required parameters
    all_params = set(config_cls.hyperparameters())
    required = {
        key for key, has_default in config_cls.hyperparameters().items() if not has_default
    }
    assert all(r in kwargs for r in required), "Keyword arguments missing at least one parameter."

    return config_cls(**{k: v for k, v in kwargs.items() if k in all_params})
