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


# Standard library imports
import os
import tempfile
from pathlib import Path
import subprocess
from shutil import copyfile

# Third-party imports
import pytest

# First-party imports
import gluonts

docker_images_path = (
    Path(gluonts.__file__).parent
    / "nursery"
    / "sagemaker_sdk"
    / "docker_images"
)
cpu_training_docker_build_makefile_path = (
    docker_images_path / "cpu_training" / "Makefile"
)
cpu_serving_docker_build_makefile_path = (
    docker_images_path / "cpu_serving" / "Makefile"
)


@pytest.mark.xfail(
    reason="The sagemaker-mxnet-container library might have changed breaking the docker makefile."
)
@pytest.mark.parametrize(
    "docker_build_makefile_path, fetch_dependency_command",
    [
        (
            cpu_training_docker_build_makefile_path,
            "sagemaker-mxnet-container_dependency",
        ),
        (
            cpu_serving_docker_build_makefile_path,
            "sagemaker-mxnet-serving-container_dependency",
        ),
    ],
)
def test_make_pre_and_post_build_tasks(
    docker_build_makefile_path, fetch_dependency_command
):
    # we need to write some data for this test, so we use a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        copyfile(docker_build_makefile_path, str(temp_dir_path / "Makefile"))

        try:
            subprocess.check_output(
                ["make", fetch_dependency_command], cwd=temp_dir_path
            )
        except subprocess.CalledProcessError as e:
            # The error message is not going to be useful in case of an error, so adding custom one:
            raise AssertionError(
                f"Something went wrong when fetching the sagemaker-mxnet-container dependency to building the "
                f"container. Check sagemaker_sdk/cpu_training and run 'make' to debug."
                f"The original error message: {e.output} "
            )

        try:
            subprocess.check_output(["make", "clean"], cwd=temp_dir_path)
        except subprocess.CalledProcessError as e:
            # The error message is not going to be useful in case of an error, so adding custom one:
            raise AssertionError(
                f"Something went wrong when cleaning up after container building. Check sagemaker_sdk/cpu_training "
                f"and run 'make' to debug. The original error message: {e.output} "
            )
