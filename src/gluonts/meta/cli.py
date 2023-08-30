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

try:
    import click
except ImportError:
    import sys

    print("`gluonts.info` requires `click` to be installed.", file=sys.stderr)
    sys.exit(1)

import gluonts


@click.group()
def cli():
    pass


@cli.command()
def version():
    click.echo(gluonts.__version__)


if __name__ == "__main__":
    cli()
