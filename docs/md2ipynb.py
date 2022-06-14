import argparse
import sys
import time
from itertools import chain
from pathlib import Path

import nbformat
import notedown
from nbclient import NotebookClient


def convert(path, kernel_name=None, timeout=40 * 60):
    with path.open() as in_file:
        notebook = notedown.MarkdownReader().read(in_file)

    print(f"=== {path.name} ", end="")
    sys.stdout.flush()

    start = time.time()

    client = NotebookClient(
        notebook,
        timeout=600,
        kernel_name=kernel_name,
        resources={'metadata': {'path': '.'}}
    )
    client.execute()

    print(f"finished evaluation in {time.time() - start} sec")

    # need to add language info to for syntax highlight
    notebook["metadata"].update(language_info={"name": "python"})

    with path.with_suffix(".ipynb").open("w") as out_file:
        out_file.write(nbformat.writes(notebook))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-k', '--kernel', dest='kernel_name', default=None, help='name of ipython kernel to use'
    )
    parser.add_argument(
        'files', type=str, nargs='+', help='path to files to convert'
    )

    args = parser.parse_args()

    here = Path(".")
    files = list(chain.from_iterable(map(here.glob, args.files)))

    for file in files:
        convert(file, kernel_name=args.kernel_name)
