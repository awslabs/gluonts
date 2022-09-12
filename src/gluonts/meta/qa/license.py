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

# Standard library imports
import sys
from itertools import filterfalse, islice
from pathlib import Path

# Third-party imports
from textwrap import dedent
from typing import Iterator, List

import click

license = dedent(
    """
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
    """
).strip()


external_license = "# LICENSE: EXTERNAL"


def check_file(path: Path) -> bool:
    """Check whether `path` is complicit.

    The license header needs to start on the 3rd line latest. This allows
    to have a shebang on top of the file.
    """
    num_lines = len(license.splitlines()) + 2

    with path.open() as in_file:
        lines = islice(in_file, num_lines)

        # lines contain `\n` character, so no need to add another one
        head = "".join(lines)

    return license in head or external_license in head


def iter_paths(roots: List[str]) -> Iterator[Path]:
    for root in map(Path, roots):
        if root.is_file():
            yield root
        else:
            if not (root / ".no-license-check").exists():
                yield from root.glob("*.py")
                yield from iter_paths(
                    subpath for subpath in root.iterdir() if subpath.is_dir()
                )


@click.group()
def cli() -> None:
    pass


@cli.command()
@click.argument("roots", type=click.Path(), nargs=-1, required=True)
def fix(roots: List[str]) -> None:
    for file_path in filterfalse(check_file, iter_paths(roots)):
        with file_path.open() as in_file:
            file_content = in_file.read()

        with file_path.open("w") as out_file:
            print(f"Prepending missing header to {file_path}")
            out_file.write(license)
            out_file.write("\n\n")
            out_file.write(file_content)


@cli.command()
@click.argument("roots", type=click.Path(), nargs=-1, required=True)
def check(roots: List[str]) -> None:
    exit_code = 0

    for file_path in filterfalse(check_file, iter_paths(roots)):
        print(f"File {file_path} does not have correct header")
        exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    cli()
