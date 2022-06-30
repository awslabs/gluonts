import argparse
import sys
import time
from itertools import chain
from pathlib import Path

import click
import jinja2
import nbformat
import notedown
from nbclient import NotebookClient

from jinja2 import Environment

env = Environment()


def convert(path, mode, kernel_name=None, timeout=40 * 60):
    with path.open() as in_file:
        template = env.from_string(in_file.read())

    notebook_text = template.render(mode=mode)
    notebook = notedown.MarkdownReader().reads(notebook_text)

    print(f"=== {path.name} ", end="")
    sys.stdout.flush()

    start = time.time()

    client = NotebookClient(
        notebook,
        timeout=600,
        kernel_name=kernel_name,
        resources={"metadata": {"path": "."}},
    )
    client.execute()

    print(f"finished evaluation in {time.time() - start} sec")

    # need to add language info to for syntax highlight
    notebook["metadata"].update(language_info={"name": "python"})

    # XXX.md.input -> XXX.ipynb
    # `with_suffix` only operates on last suffix, so we need some more involved
    # logic.
    stem = path.name.split(".", 1)[0]
    nbformat.write(notebook, path.with_name(stem).with_suffix(".ipynb"))


@click.command()
@click.argument("files", type=click.Path(), nargs=-1)
@click.option("--kernel", "-k", help="Name of iPython kernel to use.")
@click.option("--mode", "-m", default="RELEASE")
def cli(files, kernel, mode):

    for file in chain.from_iterable(map(Path().glob, files)):
        convert(file, kernel_name=kernel, mode=mode)


if __name__ == "__main__":
    cli()
