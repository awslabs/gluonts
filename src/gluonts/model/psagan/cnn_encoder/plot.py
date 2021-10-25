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

import pathlib
from pathlib import Path

import matplotlib.pyplot as plt

from gluonts.core.component import validated
from gluonts.model.psagan.cnn_encoder.helpers import EMA


class PlotterSaver:
    """
    *save_dir: path where to save the plots
    *fontsize: size of the font in the plots
    *EMA_value: value with which the exponential
    moving average is computed, if 0, it is not computed.
    """

    @validated()
    def __init__(
        self,
        save_dir: pathlib.PosixPath,
        fontsize: int = 10,
        EMA_value: float = 0.2,
    ):
        self.save_dir = save_dir
        self.EMA_value = EMA_value
        # Created nested directory if it does not already exist
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        plt.rcParams.update({"font.size": fontsize})

    def save_plot(
        self, loss, train: bool = True, alpha: float = 0.2, prefix: str = ""
    ):
        if train:
            title = "Training"
            file_name = "training_loss.png"
        else:
            title = "Validation"
            file_name = "validation_loss.png"
        file_name = prefix + "_" + file_name
        plt.plot(loss, label="Loss")
        plt.plot(EMA(loss, alpha=alpha), label="EMA loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title(title + " Loss values")
        plt.legend()
        plt.savefig(self.save_dir / file_name)
        plt.close()

    def log_loss(self, value: float, train: bool = True, prefix: str = ""):
        if train:
            file_name = "training_loss.txt"
        else:
            file_name = "validation_loss.txt"
        file_name = prefix + "_" + file_name
        with open(self.save_dir / file_name, "a") as f:
            f.write(str(value) + "\n")
            f.close()
