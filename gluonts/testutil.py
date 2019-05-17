# Standard library imports
import shutil
import tempfile
from contextlib import contextmanager


@contextmanager
def TemporaryDirectory():
    name = tempfile.mkdtemp()
    try:
        yield name
    finally:
        shutil.rmtree(name)


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i : i + n]
