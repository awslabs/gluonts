import distutils.cmd
import sys
from pathlib import Path
from textwrap import dedent

from setuptools import setup

ROOT = Path(__file__).parent


def get_version_cmdclass(version_file):
    with open(version_file) as fobj:
        code = fobj.read()

    globals_ = {"__file__": str(version_file)}
    exec(code, globals_)

    return globals_["cmdclass"]()


class TypeCheckCommand(distutils.cmd.Command):
    """A custom command to run MyPy on the project sources."""

    description = "run MyPy on Python source files"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        # import here (after the setup_requires list is loaded),
        # otherwise a module-not-found error is thrown
        import mypy.api

        mypy_opts = [
            "--allow-redefinition",
            "--follow-imports=silent",
            "--ignore-missing-imports",
        ]

        folders = [
            str(p.parent.resolve()) for p in ROOT.glob("src/**/.typesafe")
        ]

        print(
            "The following folders contain a `.typesafe` marker file "
            "and will be type-checked with `mypy`:"
        )
        for folder in folders:
            print(f"  {folder}")

        std_out, std_err, exit_code = mypy.api.run(mypy_opts + folders)

        print(std_out, file=sys.stdout)
        print(std_err, file=sys.stderr)

        if exit_code:
            error_msg = dedent(
                f"""
                Mypy command

                    mypy {" ".join(mypy_opts + folders)}

                returned a non-zero exit code. Fix the type errors listed above
                and then run

                    python setup.py type_check

                in order to validate your fixes.
                """
            ).lstrip()

            print(error_msg, file=sys.stderr)
            sys.exit(exit_code)


def find_requirements(filename):
    with open(ROOT / "requirements" / filename) as fobj:
        return [line.rstrip() for line in fobj if not line.startswith("#")]


arrow_require = find_requirements("requirements-arrow.txt")
docs_require = find_requirements("requirements-docs.txt")
tests_require = find_requirements("requirements-test.txt")
sagemaker_api_require = find_requirements(
    "requirements-extras-sagemaker-sdk.txt"
)
shell_require = find_requirements("requirements-extras-shell.txt")
mxnet_require = find_requirements("requirements-mxnet.txt")
torch_require = find_requirements("requirements-pytorch.txt")

dev_require = (
    arrow_require
    + docs_require
    + tests_require
    + shell_require
    + sagemaker_api_require
)

setup(
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
    cmdclass={
        "type_check": TypeCheckCommand,
        **get_version_cmdclass("src/gluonts/meta/_version.py"),
    },
)
