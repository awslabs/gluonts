# Standard library imports
import distutils.cmd
import distutils.log
import itertools
import logging
import os
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

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

try:
    from sphinx import apidoc, setup_command

    HAS_SPHINX = True
except ImportError:

    logging.warning(
        "Package 'sphinx' not found. You will not be able to build the docs."
    )

    HAS_SPHINX = False


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
        mypy_args = [str(p.parent.resolve()) for p in SRC.glob("**/.typesafe")]

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
        import click
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
        except click.exceptions.Exit as e:
            exit_code = e.exit_code

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


docs_require = find_requirements("requirements-docs.txt")
tests_require = find_requirements("requirements-test.txt")
sagemaker_api_require = find_requirements(
    "requirements-extras-sagemaker-sdk.txt"
)
shell_require = find_requirements("requirements-extras-shell.txt")
setup_requires = find_requirements("requirements-setup.txt")
dev_require = (
    docs_require
    + tests_require
    + shell_require
    + setup_requires
    + sagemaker_api_require
)

setup_kwargs: dict = dict(
    name="gluonts",
    use_scm_version={"fallback_version": "0.0.0"},
    description=(
        "GluonTS is a Python toolkit for probabilistic time series modeling, "
        "built around MXNet."
    ),
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    url="https://github.com/awslabs/gluon-ts",
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
        "dev": dev_require,
        "docs": docs_require,
        "R": find_requirements("requirements-extras-r.txt"),
        "Prophet": find_requirements("requirements-extras-prophet.txt"),
        "shell": shell_require,
    },
    entry_points=dict(
        gluonts_forecasters=[
            "deepar=gluonts.model.deepar:DeepAREstimator",
            "DeepAR=gluonts.model.deepar:DeepAREstimator",
            "DeepFactor=gluonts.model.deep_factor:DeepFactorEstimator",
            "DeepState=gluonts.model.deepstate:DeepStateEstimator",
            "DeepVAR=gluonts.model.deepvar:DeepVAREstimator",
            "GaussianProcess=gluonts.model.gp_forecaster:GaussianProcessEstimator",
            "GPVAR=gluonts.model.gpvar:GPVAREstimator",
            "LSTNet=gluonts.model.lstnet:LSTNetEstimator",
            "NBEATS=gluonts.model.n_beats:NBEATSEstimator",
            "NBEATSEnsemble=gluonts.model.n_beats:NBEATSEnsembleEstimator",
            "NPTS=gluonts.model.npts:NPTSPredictor",
            "Rotbaum=gluonts.model.rotbaum:TreeEstimator",
            "SelfAttention=gluonts.model.san:SelfAttentionEstimator",
            "SeasonalNaive=gluonts.model.seasonal_naive:SeasonalNaivePredictor",
            "MQCNN=gluonts.model.seq2seq:MQCNNEstimator",
            "MQRNN=gluonts.model.seq2seq:MQRNNEstimator",
            "Seq2Seq=gluonts.model.seq2seq:Seq2SeqEstimator",
            "SimpleFeedForward=gluonts.model.simple_feedforward:SimpleFeedForwardEstimator",
            "TFT=gluonts.model.tft:TemporalFusionTransformerEstimator",
            "DeepTPP=gluonts.model.tpp:DeepTPPEstimator",
            "Transformer=gluonts.model.transformer:TransformerEstimator",
            "Constant=gluonts.model.trivial.constant:ConstantPredictor",
            "ConstantValue=gluonts.model.trivial.constant:ConstantValuePredictor",
            "Identity=gluonts.model.trivial.identity:IdentityPredictor",
            "Mean=gluonts.model.trivial.mean:MeanEstimator",
            "MeanPredictor=gluonts.model.trivial.mean:MeanPredictor",
            "MovingAverage=gluonts.model.trivial.mean:MovingAveragePredictor",
            "WaveNet=gluonts.model.wavenet:WaveNetEstimator",
            # "r=gluonts.model.r_forecast:RForecastPredictor [R]",
            # "prophet=gluonts.model.prophet:ProphetPredictor [Prophet]",
        ]
    ),
    cmdclass={
        "type_check": TypeCheckCommand,
        "style_check": StyleCheckCommand,
    },
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

    for command in ["build_sphinx", "doc", "docs"]:
        setup_kwargs["cmdclass"][command] = BuildApiDoc

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
