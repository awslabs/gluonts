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
`gluonts.shell.sagemaker.dyn`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dynamic code loading for `gluonts.shell`.
"""


import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from functools import partial
from pathlib import Path
from typing import Optional


class Installer:
    def __init__(self, packages):
        self.packages = packages
        self.cleanups = []

    def cleanup(self):
        for cleanup in self.cleanups:
            cleanup()

    def copy_install(self, path: Path):
        if path.is_file():
            shutil.copy(path, self.packages)
        elif path.is_dir():
            shutil.copytree(path, self.packages / path.name)

    def pip_install(self, path: Path):
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--upgrade",
                "--target",
                str(self.packages),
                str(path),
            ]
        )

    def install(self, path):
        if path.is_file():
            if tarfile.is_tarfile(path):
                self.handle_archive(tarfile.open, path)

            elif zipfile.is_zipfile(path):
                self.handle_archive(zipfile.ZipFile, path)

            elif path.suffix == ".py":
                self.copy_install(path)

        elif path.is_dir():
            if (path / "setup.py").exists():
                self.pip_install(path)
            elif (path / "__init__.py").exists():
                self.copy_install(path)
            else:
                for subpath in path.iterdir():
                    self.install(subpath)

    def handle_archive(self, open_fn, path):
        with open_fn(path) as archive:
            tempdir = tempfile.mkdtemp()
            self.cleanups.append(
                partial(shutil.rmtree, tempdir, ignore_errors=True)
            )

            archive.extractall(tempdir)
            self.install(Path(tempdir))


def install_and_restart(code_channel: Optional[Path], packages: Path):
    # skip if there is no code, and if we already done our work
    if code_channel is None or "__SHELL_RELOADED__" in os.environ:
        return

    packages.mkdir(exist_ok=True)

    inst = Installer(packages)
    inst.install(code_channel)
    inst.cleanup()

    python_path = os.environ.get("PYTHONPATH", "")
    python_path = f"{packages}:{python_path}"

    os.environ.update(__SHELL_RELOADED__="1", PYTHONPATH=python_path)

    # restart
    os.execve(
        sys.executable,
        [sys.executable, "-m", "gluonts.shell"] + sys.argv[1:],
        os.environ,
    )
