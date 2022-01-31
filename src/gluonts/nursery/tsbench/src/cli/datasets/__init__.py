from ._main import datasets
from .compute_catch22 import compute_catch22  # type: ignore
from .compute_stats import compute_stats  # type: ignore
from .download import download  # type: ignore
from .upload import upload  # type: ignore

__all__ = ["datasets"]
