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

from pytorch_lightning import Callback, Trainer, LightningModule


class ParameterCountCallback(Callback):  # type: ignore
    """
    This callback allows counting model parameters during training.
    The output is printed to the console and can be retrieved from the log files.
    """

    def __init__(self) -> None:
        super().__init__()

    def on_pretrain_routine_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        # compute number of model params
        model_total_params = sum(
            p.numel() for p in pl_module.model.parameters()
        )
        model_total_trainable_params = sum(
            p.numel() for p in pl_module.model.parameters() if p.requires_grad
        )

        # log
        print("\n" + f"model_total_params: {model_total_params},")
        print(
            "\n"
            + f"model_total_trainable_params: {model_total_trainable_params},"
        )
