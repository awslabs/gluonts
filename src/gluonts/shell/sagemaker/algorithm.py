import shutil
import subprocess
import sys
from typing import List, Optional

import pydantic

from gluonts.shell import log


class AlgorithmConfig(pydantic.BaseModel):
    forecaster: str
    pre: Optional[List[str]] = []

    def run_pre(self, cwd):
        for command in self.pre:
            log.info("Running pre step %s", command)
            output = subprocess.check_output(command, shell=True, cwd=cwd)
            for line in output.decode("utf-8").splitlines():
                log.info("> %s", line)


def load(algo_path, is_train=False):
    sys.path.insert(0, str(algo_path))

    config = AlgorithmConfig.parse_file(algo_path / "algorithm.json")
    log.info(
        "Loading forecaster from algorithm channel: %s", config.forecaster
    )

    config.run_pre(algo_path)

    if is_train:
        target = algo_path.parents[2] / "model" / "algorithm"
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(algo_path, target)

    return config.forecaster
