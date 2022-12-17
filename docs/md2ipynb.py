import json
import re
import sys
import time
import os
from itertools import chain
from pathlib import Path

import black
import click
import jinja2
import nbformat
import notedown
from nbclient import NotebookClient

from jinja2 import Environment

env = Environment()


def check_github_event(default):
    if "GITHUB_EVENT_PATH" not in os.environ:
        return default

    with open(os.environ["GITHUB_EVENT_PATH"]) as infile:
        event = json.load(infile)

    if "pull_request" in event:
        for label in event["pull_request"]["labels"]:
            if label["name"] == "pr:docs-build-notebook":
                return default

        return "skip"

    return default


def run_notebook(text, kernel_name, timeout) -> str:
    # We add two blank lines at the end to ensure
    # that the final cell also runs.
    text += "\n" * 2
    notebook = notedown.MarkdownReader().reads(text)

    kwargs = {}
    if kernel_name is not None:
        kwargs["kernel_name"] = kernel_name

    client = NotebookClient(
        notebook,
        timeout=timeout,
        resources={"metadata": {"path": "."}},
        **kwargs,
    )
    client.execute()

    # need to add language info to for syntax highlight
    notebook["metadata"].update(language_info={"name": "python"})

    return nbformat.writes(notebook)


def black_cells(text):
    CODE_RE = r"```py(?:thon)?\s*\n(.*?)```"

    text = re.sub(r"^%", r"#%#", text, flags=re.M)

    def apply_black(match):
        code = match.group(1)

        formatted = black.format_str(code, mode=black.Mode())

        return "\n".join(["```", formatted.rstrip(), "```"])

    formatted = re.sub(CODE_RE, apply_black, text, flags=re.S)
    return re.sub(r"^#%#", r"%", formatted, flags=re.M)


def convert(path, mode, kernel_name=None, timeout=40 * 60):
    print(f"=== {path.name} ", end="")
    sys.stdout.flush()

    with path.open() as in_file:
        template = env.from_string(in_file.read())

    markdown = template.render(mode=mode)
    markdown = black_cells(markdown)

    if mode != "skip":
        suffix = ".ipynb"
        start = time.time()
        output = run_notebook(markdown, kernel_name, timeout)
        print(f"finished evaluation in {time.time() - start} sec")
    else:
        suffix = ".md"
        print(f"convert to {suffix}")
        output = markdown

    # XXX.md.input -> XXX.ipynb
    # `with_suffix` only operates on last suffix, so we need some more involved
    # logic.
    stem = path.name.split(".", 1)[0]
    with path.with_name(stem).with_suffix(suffix).open("w") as outfile:
        outfile.write(output)


@click.command()
@click.argument("paths", type=click.Path(), nargs=-1)
@click.option("--kernel", "-k", help="Name of iPython kernel to use.")
@click.option("--mode", "-m", default="release")
def cli(paths, kernel, mode):
    mode = check_github_event(mode)

    for file in map(Path, paths):
        convert(file, kernel_name=kernel, mode=mode)


if __name__ == "__main__":
    cli()
