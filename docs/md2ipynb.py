import sys
import time
from itertools import chain
from pathlib import Path

import nbformat
import notedown


def convert(path, timeout=40 * 60):
    with path.open() as in_file:
        notebook = notedown.MarkdownReader().read(in_file)

    start = time.time()
    notedown.run(notebook, timeout)

    print(f"=== {path.name} finished evaluation in {time.time() - start} sec")

    # need to add language info to for syntax highlight
    notebook["metadata"].update(language_info={"name": "python"})

    with path.with_suffix(".ipynb").open("w") as out_file:
        out_file.write(nbformat.writes(notebook))


if __name__ == "__main__":
    assert len(sys.argv) >= 2, "usage: input.md"

    here = Path(".")

    files = list(chain.from_iterable(map(here.glob, sys.argv[1:])))

    for file in files:
        convert(file)
