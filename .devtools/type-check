#!/usr/bin/env python3

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

import sys
from pathlib import Path

import click
import mypy.api

mypy_opts = [
    "--allow-redefinition",
    "--follow-imports=silent",
    "--ignore-missing-imports",
]


@click.command()
@click.argument("path", type=click.Path(file_okay=False, exists=True))
def run_mypy(path):
    directory = Path(path)

    # only run on folders containing a `.typesafe` marker file

    folders = [
        str(marker.parent) for marker in directory.rglob("**/.typesafe")
    ]

    if not folders:
        click.secho("No folders with .typesafe marker found.", fg="red")
        sys.exit(1)

    std_out, std_err, exit_code = mypy.api.run(mypy_opts + folders)

    print(std_out, file=sys.stdout)
    print(std_err, file=sys.stderr)

    sys.exit(exit_code)


if __name__ == "__main__":
    run_mypy()
