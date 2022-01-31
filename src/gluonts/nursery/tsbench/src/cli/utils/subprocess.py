import os
import subprocess
from pathlib import Path
from typing import Any


def run_sacred_script(script: str, **kwargs: Any) -> None:
    """
    Runs the Sacred script with the provided name in a subprocess, passing the provided parameters.

    Args:
        script: The name of the script in `cli/sacred`.
        kwargs: Parameters to pass to the script.
    """
    subprocess.call(
        [
            "poetry",
            "run",
            "python",
            "-W",
            "ignore",
            Path(os.path.realpath(__file__)).parent.parent
            / "analysis"
            / "scripts"
            / script,
            "-m",
            "localhost:27017:sacred",
            "with",
        ]
        + [f"{k}={_sacred_value(v)}" for k, v in kwargs.items()],
        cwd=Path(os.path.realpath(__file__)).parent.parent.parent.parent,
    )


def _sacred_value(value: Any) -> str:
    return f'"{value}"' if isinstance(value, str) else f"{value}"
