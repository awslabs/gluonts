import os
from typing import Any


def upload_directory(client: Any, directory: str, bucket: str, prefix: str = "") -> None:
    """
    Uploads all files in the given directory (recursively) to the provided bucket, using the
    specified prefix.

    Args:
        client: The S3 client to use for uploading.
        directory: The path to the directory to upload.
        bucket: The bucket where to upload the directory.
        prefix: The prefix to use when uploading the directory.
    """
    for root, _, files in os.walk(directory):
        for file in files:
            path = os.path.join(root, file)
            with open(path, "rb") as f:
                bucket_path = os.path.relpath(path, directory)
                client.upload_fileobj(f, bucket, f"{prefix}/{bucket_path}")
