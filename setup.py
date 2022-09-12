# Standard library imports
import distutils.cmd
import os
import subprocess
from pathlib import Path

# Third-party imports
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


def read(*names, encoding="utf8"):
    with (ROOT / Path(*names)).open(encoding=encoding) as fp:
        return fp.read()


def find_requirements(filename):
    with (ROOT / "requirements" / filename).open() as f:
        mxnet_old = "mxnet"
        mxnet_new = "mxnet-cu92mkl" if GPU_SUPPORT else mxnet_old
        return [
            line.rstrip().replace(mxnet_old, mxnet_new, 1)
            for line in f
            if not (line.startswith("#") or line.startswith("http"))
        ]


def get_version_and_cmdclass(version_file):
    with open(version_file) as fobj:
        code = fobj.read()

    globals_ = {"__file__": str(version_file)}
    exec(code, globals_)

    return globals_["__version__"], globals_["cmdclass"]()


version, version_cmdclass = get_version_and_cmdclass(
    "src/gluonts/meta/_version.py"
)


arrow_require = find_requirements("requirements-arrow.txt")
docs_require = find_requirements("requirements-docs.txt")
tests_require = find_requirements("requirements-test.txt")
sagemaker_api_require = find_requirements(
    "requirements-extras-sagemaker-sdk.txt"
)
shell_require = find_requirements("requirements-extras-shell.txt")
mxnet_require = find_requirements("requirements-mxnet.txt")
torch_require = find_requirements("requirements-pytorch.txt")

setup_requires = find_requirements("requirements-setup.txt")

dev_require = (
    arrow_require
    + docs_require
    + tests_require
    + shell_require
    + setup_requires
    + sagemaker_api_require
)

setup_kwargs: dict = dict(
    name="gluonts",
    version=version,
    description=(
        "GluonTS is a Python toolkit for probabilistic time series modeling, "
        "built around MXNet."
    ),
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    url="https://github.com/awslabs/gluonts/",
    project_urls={
        "Documentation": "https://ts.gluon.ai/stable/",
        "Source Code": "https://github.com/awslabs/gluonts/",
    },
    author="Amazon",
    author_email="gluon-ts-dev@amazon.com",
    maintainer_email="gluon-ts-dev@amazon.com",
    license="Apache License 2.0",
    python_requires=">= 3.6",
    package_dir={"": "src"},
    packages=find_namespace_packages(include=["gluonts*"], where=str(SRC)),
    include_package_data=True,
    setup_requires=setup_requires,
    install_requires=find_requirements("requirements.txt"),
    tests_require=tests_require,
    extras_require={
        "arrow": arrow_require,
        "dev": dev_require,
        "docs": docs_require,
        "mxnet": mxnet_require,
        "R": find_requirements("requirements-extras-r.txt"),
        "Prophet": find_requirements("requirements-extras-prophet.txt"),
        "pro": arrow_require + ["orjson"],
        "shell": shell_require,
        "torch": torch_require,
    },
    cmdclass=version_cmdclass,
    entry_points={
        "pygments.styles": [
            "gluonts-dark=gluonts.meta.style:Dark",
        ]
    },
)

# -----------------------------------------------------------------------------
# start of AWS-internal section (DO NOT MODIFY THIS SECTION)!
#
# all AWS-internal configuration goes here
#
# end of AWS-internal section (DO NOT MODIFY THIS SECTION)!
# -----------------------------------------------------------------------------

# do the work

if __name__ == "__main__":
    setup(**setup_kwargs)
