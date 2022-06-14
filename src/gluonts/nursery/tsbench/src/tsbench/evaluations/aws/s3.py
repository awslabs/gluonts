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

import os
from typing import Any


def upload_directory(
    client: Any, directory: str, bucket: str, prefix: str = ""
) -> None:
    """
    Uploads all files in the given directory (recursively) to the provided
    bucket, using the specified prefix.

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
