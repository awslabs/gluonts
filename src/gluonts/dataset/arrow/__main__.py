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

from pathlib import Path

import click

from gluonts.dataset.common import FileDataset
from gluonts.gluonts_tqdm import tqdm

from . import ArrowWriter, ParquetWriter


@click.group()
def cli():
    pass


@cli.command()
@click.argument("dataset", type=click.Path(exists=True))
@click.argument("out", type=click.Path())
@click.option("--freq", required=True)
@click.option("--type", "type_", type=click.Choice(["arrow", "parquet"]))
@click.option("--stream", default=False)
def write(dataset, out, freq, type_, stream):
    out = Path(out)

    if type_ is None:
        if out.suffix == ".arrow":
            type_ = "arrow"
        elif out.suffix == ".parquet":
            type_ = "parquet"
        else:
            raise click.UsageError(f"Unsupported suffix {out.suffix}.")

    if type_ == "arrow":
        writer = ArrowWriter(stream=stream)
    else:
        writer = ParquetWriter()

    writer.write_to_file(tqdm(FileDataset(dataset, freq)), out)


if __name__ == "__main__":
    cli()
