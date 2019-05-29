# Standard library imports
import distutils.cmd
import distutils.log
import itertools
import logging
import subprocess
import os  # noqa
import sys
from textwrap import dedent
from pathlib import Path

# Third-party imports
import setuptools
import setuptools.command.build_py
from setuptools import setup, find_namespace_packages

ROOT = Path(__file__).parent

VERSION = "0.1.0"

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

    with (ROOT / "gluonts" / "version.py").open("w") as f:
        f.write(content)


def find_requirements(filename):
    with (ROOT / filename).open() as f:
        mxnet_old = "mxnet=="
        mxnet_new = "mxnet-cu92mkl==" if GPU_SUPPORT else mxnet_old
        return [
            line.rstrip().replace(mxnet_old, mxnet_new, 1)
            for line in f
            if not line.startswith("#")
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
            str(p.parent.resolve()) for p in ROOT.glob("**/.typesafe")
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
            for folder in ["gluonts", "test", "examples"]
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
    description=("A toolkit for time series modeling with neural networks."),
    author="Amazon",
    author_email="gluon-ts-dev@amazon.com",
    maintainer_email="gluon-ts-dev@amazon.com",
    license="Apache License 2.0",
    python_requires=">= 3.6",
    packages=find_namespace_packages(include=["gluonts*"], where=str(ROOT)),
    include_package_data=True,
    setup_requires=find_requirements("requirements-setup.txt"),
    install_requires=find_requirements("requirements.txt"),
    tests_require=find_requirements("requirements-test.txt"),
    entry_points=dict(
        console_scripts=[
            "gluonts-validate-dataset=gluonts.dataset.validate:run"
        ]
    ),
    test_flake8=True,
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
                    [str(ROOT)],  # in path
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

# update the public setup arguments
if 'BRAZIL_PACKAGE_NAME' in os.environ:
    setup_kwargs = {
        **setup_kwargs,
        **dict(
            # Brazil does not play well with requirements lists - reset them
            setup_requires=[],
            install_requires=[],
            tests_require=[],
            options=dict(
                # make sure the right shebang is set for the scripts -
                # use the environment default Python 3.6
                build_scripts=dict(
                    executable='/apollo/sbin/envroot "$ENVROOT/bin/python3.6"'
                )
            ),
            # set to the name of the interpreter whose scripts should go
            # in $ENVROOT/bin
            # see: https://w.amazon.com/index.php/BrazilPython3
            root_script_source_version="python3.6",
        ),
    }

# override build_sphinx command to a Brazil-compatible version
if 'BRAZIL_PACKAGE_NAME' in os.environ and HAS_SPHINX:
    index_content = dedent(
        """
        <html>
        <head>
          <meta http-equiv="REFRESH" content="0; url=html/index.html">
        </head>
        <body>
          Redirecting to Sphinx documentation:
          <a href="html/index.html">
            Click here if not redirected automatically
          </a>
        </body>
        </html>
        """
    ).lstrip()

    class BrazilBuildApiDoc(BuildApiDoc):
        def initialize_options(self):
            super().initialize_options()
            brazil_src_dir = os.getcwd()
            while brazil_src_dir != '/' and brazil_src_dir != '':
                if os.path.exists(os.path.join(brazil_src_dir, 'Config')):
                    break
                brazil_src_dir = os.path.dirname(brazil_src_dir)
            if brazil_src_dir == '/' or brazil_src_dir == '':
                raise RuntimeError('Unable to find Brazil source directory.')
            self.build_dir = os.path.join(
                brazil_src_dir, 'build', 'brazil-documentation'
            )

        def run(self):
            super(BrazilBuildApiDoc, self).run()
            # Write redirect page to 'index.html' file in 'html' directory.
            with (Path(self.build_dir) / "index.html").open("w") as f:
                f.write(index_content)

    for command in ['build_sphinx', 'doc', 'docs']:
        setup_kwargs["cmdclass"][command] = BrazilBuildApiDoc

# end of AWS-internal section (DO NOT MODIFY THIS SECTION)!
# -----------------------------------------------------------------------------

# do the work
write_version_py()
setup(**setup_kwargs)
