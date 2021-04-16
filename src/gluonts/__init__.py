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

# !!! DO NOT MODIFY !!! (pkgutil-style namespace package)

from pkgutil import extend_path


from functools import partial
from subprocess import check_output, CalledProcessError, PIPE


def _get_version():
    def run(command):
        return check_output(
            command.split(), stderr=PIPE, encoding="utf-8"
        ).strip()

    try:
        # if HEAD is tagged, we use that as the version
        return run("git describe --exact-match --tags")
    except CalledProcessError:
        pass

    try:
        commit_id = run("git rev-parse --short HEAD")
        branch = run("git rev-parse --abbrev-ref HEAD")
        return f"dev.{branch}+g{commit_id}"
    except CalledProcessError:
        return "0.0.0-unknown"


__path__ = extend_path(__path__, __name__)  # type: ignore
__version__ = _get_version()
