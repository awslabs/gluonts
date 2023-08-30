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

from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning import Trainer
import numpy as np
from pathlib import Path


def get_save_dir_from_csvlogger(logger: CSVLogger) -> Path:
    return Path(logger.save_dir) / logger.name / f"version_{logger.version}"


def get_loss_steps(loss_name: str, trainer: Trainer):
    loss = trainer.logger.experiment.metrics
    ep = trainer.current_epoch
    out = [l[loss_name] for l in loss if loss_name in l]
    start = 1 if "val" in loss_name else 0
    return out, np.linspace(start, ep, len(out))
