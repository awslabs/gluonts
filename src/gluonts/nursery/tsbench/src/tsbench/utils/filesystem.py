import os
import tarfile
from pathlib import Path
from typing import Optional, Set


def compress_directory(
    directory: Path, target: Path, include: Optional[Set[str]] = None
) -> None:
    """
    Compresses the provided directory into a single `.tar.gz` file.

    Args:
        directory: The directory to compress.
        target: The `.tar.gz` file where the compressed archive should be written.
        include: The filenames to include. If not provided, all files are included.
    """
    with target.open("wb+") as f:
        with tarfile.open(fileobj=f, mode="w:gz") as tar:
            for root, _, files in os.walk(directory):
                for file in files:
                    if include is not None and file not in include:
                        continue
                    name = os.path.join(root, file)
                    tar.add(name, arcname=os.path.relpath(name, directory))
