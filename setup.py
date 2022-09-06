from itertools import chain
from pathlib import Path
from setuptools import find_namespace_packages, setup

ROOT = Path(__file__).parent
SRC = ROOT / "src"


def find_requirements(name=None):
    if name:
        filename = f"requirements-{name}.txt"
    else:
        filename = "requirements.txt"

    with open(ROOT / "requirements" / filename) as req_file:
        return [
            line.rstrip()
            for line in req_file
            if not line.startswith("#") and not line.startswith("http")
        ]


def get_version_and_cmdclass(version_file):
    with open(version_file) as fobj:
        code = fobj.read()

    globals_ = {"__file__": str(version_file)}
    exec(code, globals_)

    return globals_["__version__"], globals_["cmdclass"]()


version, version_cmdclass = get_version_and_cmdclass("src/gluonts/_version.py")


optional_dependencies = {
    "arrow": find_requirements("arrow"),
    "docs": find_requirements("docs"),
    "test": find_requirements("test"),
    "mxnet": find_requirements("mxnet"),
    "torch": find_requirements("pytorch"),
    "shell": find_requirements("extras-shell"),
    "R": find_requirements("extras-r"),
    "Prophet": find_requirements("extras-prophet"),
    "sagemaker-sdk": find_requirements("extras-sagemaker-sdk"),
}

# `dev` includes all requirements
optional_dependencies["dev"] = list(
    chain.from_iterable(optional_dependencies.values())
)


setup(
    name="gluonts",
    version=version,
    package_dir={"": "src"},
    packages=find_namespace_packages(include=["gluonts*"], where=str(SRC)),
    include_package_data=True,
    install_requires=find_requirements(),
    tests_require=optional_dependencies["test"],
    extras_require=optional_dependencies,
    cmdclass=version_cmdclass,
)
