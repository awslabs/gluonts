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

import json
from typing import Dict, Any
from pytorch_lightning import Callback, Trainer, LightningModule

from .common import get_save_dir_from_csvlogger


class InitialSaveCallback(Callback):  # type: ignore
    """
    This callback saves the initial model using the save method of the LightKit ConfigModule.
    This allows to load the model later without access to the hyper parameters needed to instantiate the class.
    Additionally, a dictionary of arguments is stored.

    Args:
        args_to_save: Contains the arguments that are stored.
    """

    def __init__(self, args_to_save: Dict[str, Any]) -> None:
        super().__init__()
        self.args_to_save = args_to_save

    def on_pretrain_routine_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        base_dir = get_save_dir_from_csvlogger(trainer.logger)
        # save the dictionary
        with open(base_dir / "args.json", "w") as fp:
            json.dump(self.args_to_save, fp)

        # save model
        save_dir = base_dir / "initial_model"
        save_dir.mkdir(parents=True, exist_ok=True)
        pl_module.model.save(path=save_dir)
