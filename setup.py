# Standard library imports
import distutils.cmd
import distutils.log
import io
import itertools
import logging
import re
import subprocess
import os  # noqa
import sys
from pathlib import Path
from textwrap import dedent

# Third-party imports
import setuptools
import setuptools.command.build_py
from setuptools import find_namespace_packages, setup

ROOT = Path(__file__).parent
SRC = ROOT / 'src'


def read(*names, **kwargs):
    with io.open(
        os.path.join(os.path.dirname(__file__), *names),
        encoding=kwargs.get("encoding", "utf8"),
    ) as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(
        r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M
    )
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


VERSION = find_version('src', 'gluonts', '__init__.py')

GPU_SUPPORT = 0 == int(
    subprocess.call(
        "nvidia-smi",
        shell=True,
        stdout=open("/dev/null", "w"),
        stderr=open("/dev/null", "w"),
    )
)

try:
    from sphinx import apidoc, setup_command

    HAS_SPHINX = True
except ImportError:

    logging.warning(
        "Package 'sphinx' not found. You will not be able to build the docs."
    )

    HAS_SPHINX = False


def get_git_hash():
    try:
        sp = subprocess.Popen(
            ["git", "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        hash = sp.communicate()[0].decode("utf-8").strip()
        sp = subprocess.Popen(
            ["git", "diff-index", "--quiet", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        sp.communicate()
        if sp.returncode != 0:
            return "modified (latest: {})".format(hash)
        else:
            return hash
    except:  # noqa
        return "unkown"


def write_version_py():
    content = dedent(
        f"""
        # This file is auto generated. Don't modify it and don't add it to git.
        PKG_VERSION = '{VERSION}'
        GIT_REVISION = '{get_git_hash()}'
        """
    ).lstrip()

    with (SRC / "gluonts" / "version.py").open("w") as f:
        f.write(content)


def find_requirements(filename):
    with (ROOT / filename).open() as f:
        mxnet_old = "mxnet=="
        mxnet_new = "mxnet-cu92mkl==" if GPU_SUPPORT else mxnet_old
        return [
            line.rstrip().replace(mxnet_old, mxnet_new, 1)
            for line in f
            if not (line.startswith("#") or line.startswith("http"))
        ]


class TypeCheckCommand(distutils.cmd.Command):
    """A custom command to run MyPy on the project sources."""

    description = "run MyPy on Python source files"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        """Run command."""

        # import here (after the setup_requires list is loaded),
        # otherwise a module-not-found error is thrown
        import mypy.api

        mypy_opts = ["--follow-imports=silent", "--ignore-missing-imports"]
        mypy_args = [
            str(p.parent.resolve()) for p in SRC.glob("**/.typesafe")
        ]

        print(
            "the following folders contain a `.typesafe` marker file "
            "and will be type-checked with `mypy`:"
        )
        print("\n".join(["  " + arg for arg in mypy_args]))

        std_out, std_err, exit_code = mypy.api.run(mypy_opts + mypy_args)

        print(std_out, file=sys.stdout)
        print(std_err, file=sys.stderr)

        if exit_code:
            error_msg = dedent(
                f"""
                Mypy command

                    mypy {" ".join(mypy_opts + mypy_args)}

                returned a non-zero exit code. Fix the type errors listed above
                and then run

                    python setup.py type_check

                in order to validate your fixes.
                """
            ).lstrip()

            print(error_msg, file=sys.stderr)
            sys.exit(exit_code)


class StyleCheckCommand(distutils.cmd.Command):
    """A custom command to run MyPy on the project sources."""

    description = "run Black style check on Python source files"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        """Run command."""

        # import here (after the setup_requires list is loaded),
        # otherwise a module-not-found error is thrown
        import black

        black_opts = []
        black_args = [
            str(ROOT / folder)
            for folder in ["src", "test", "examples"]
            if (ROOT / folder).is_dir()
        ]

        print(
            "Python files in the following folders will be style-checked "
            "with `black`:"
        )
        print("\n".join(["  " + arg for arg in black_args]))

        # a more direct way to call black
        # this bypasses the problematic `_verify_python3_env` call in
        # `click.BaseCommand.main`, which brakes things on Brazil builds
        ctx = black.main.make_context(
            info_name="black", args=["--check"] + black_opts + black_args
        )
        try:
            exit_code = black.main.invoke(ctx)
        except SystemExit as e:
            exit_code = e.code

        if exit_code:
            error_msg = dedent(
                f"""
                Black command

                    black {" ".join(['--check'] + black_opts + black_args)}

                returned a non-zero exit code. Fix the files listed above with

                    black {" ".join(black_opts + black_args)}

                and then run

                    python setup.py style_check

                in order to validate your fixes.
                """
            ).lstrip()

            print(error_msg, file=sys.stderr)
            sys.exit(exit_code)


class BuildPyCommand(setuptools.command.build_py.build_py):
    """Custom `build_py` command that preprends a `typecheck` call."""

    def run(self):
        self.run_command("type_check")
        self.run_command("style_check")
        setuptools.command.build_py.build_py.run(self)


setup_kwargs: dict = dict(
    name="gluonts",
    version=VERSION,
    description=(
        "GluonTS is a Python toolkit for probabilistic time series modeling, "
        "built around MXNet."
    ),
    long_description=read('README.md'),
    long_description_content_type='text/markdown',
    url='https://github.com/awslabs/gluon-ts',
    author="Amazon",
    author_email="gluon-ts-dev@amazon.com",
    maintainer_email="gluon-ts-dev@amazon.com",
    license="Apache License 2.0",
    python_requires=">= 3.6",
    package_dir={'':'src'},
    packages=find_namespace_packages(include=["gluonts*"], where=str(SRC)),
    include_package_data=True,
    setup_requires=find_requirements("requirements-setup.txt"),
    install_requires=find_requirements("requirements.txt"),
    tests_require=find_requirements("requirements-test.txt"),
    extras_require={
        'R': find_requirements("requirements-extras-r.txt"),
        'Prophet': find_requirements("requirements-extras-prophet.txt"),
    },
    entry_points=dict(
        console_scripts=[
            "gluonts-validate-dataset=gluonts.dataset.validate:run"
        ]
    ),
    cmdclass=dict(
        type_check=TypeCheckCommand,
        style_check=StyleCheckCommand,
        build_py=BuildPyCommand,
    ),
)

if HAS_SPHINX:

    class BuildApiDoc(setup_command.BuildDoc):
        def run(self):
            args = list(
                itertools.chain(
                    ["-f"],  # force re-generation
                    ["-P"],  # include private modules
                    ["--implicit-namespaces"],  # respect PEP420
                    ["-o", str(ROOT / "docs" / "api" / "gluonts")],  # out path
                    [str(SRC / "gluonts")],  # in path
                    ["setup*", "test", "docs", "*pycache*"],  # excluded paths
                )
            )
            apidoc.main(args)
            super(BuildApiDoc, self).run()

    setup_kwargs["doc_command"] = "build_sphinx"
    for command in ['build_sphinx', 'doc', 'docs']:
        setup_kwargs["cmdclass"][command] = BuildApiDoc

# -----------------------------------------------------------------------------
# start of AWS-internal section (DO NOT MODIFY THIS SECTION)!
#
# all AWS-internal configuration goes here
#
# end of AWS-internal section (DO NOT MODIFY THIS SECTION)!
# -----------------------------------------------------------------------------

# do the work
write_version_py()
setup(**setup_kwargs)
