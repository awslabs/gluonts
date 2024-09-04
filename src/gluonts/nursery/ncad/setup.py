import os
from pathlib import Path
import subprocess

from setuptools import find_namespace_packages, setup

ROOT = Path(__file__).parent
SRC = ROOT / "src"
GPU_SUPPORT = 0 == int(
    subprocess.call(
        "nvidia-smi",
        shell=True,
        stdout=open(os.devnull, "w"),
        stderr=open(os.devnull, "w"),
    )
)

def read(*names, encoding:str="utf8"):
    with (ROOT / Path(*names)).open(encoding=encoding) as fp:
        return fp.read()

def find_requirements(filename:str):
    with (ROOT / "requirements" / filename).open() as f:
        return [
            line.rstrip()
            for line in f
            if not (line.startswith("#") or line.startswith("http"))
        ]

install_requires = find_requirements("requirements.txt")
setup_requires = find_requirements("requirements-setup.txt")
docs_require = find_requirements("requirements-docs.txt")
tests_require = find_requirements("requirements-test.txt")
examples_require = find_requirements("requirements-examples.txt")
dev_require = (
    docs_require
    + setup_requires
    + tests_require
    + examples_require
)


setup_kwargs: dict = dict(
    name="ncad",
    
    use_scm_version={"fallback_version": "0.0.0"},

    description="Neural Contextual Anomaly Detection for Time Series",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    url="https://github.com/awslabs/gluon-ts",
    license="Apache License 2.0",
    python_requires=">= 3.6, < 3.9",
    package_dir={"": "src"},
    packages=find_namespace_packages(include=["ncad*"], where=str(SRC)),
    include_package_data=True,
    setup_requires=setup_requires,
    install_requires=install_requires,
    tests_require=tests_require,
    extras_require={
        "dev": dev_require,
        "docs": docs_require,
        "examples": examples_require,
    },
)

if __name__ == "__main__":
    setup(**setup_kwargs)
