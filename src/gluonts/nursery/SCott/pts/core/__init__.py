# Relative imports
from ._base import fqname_for

__all__ = ["fqname_for"]

# fix Sphinx issues, see https://bit.ly/2K2eptM
for item in __all__:
    if hasattr(item, "__module__"):
        setattr(item, "__module__", __name__)