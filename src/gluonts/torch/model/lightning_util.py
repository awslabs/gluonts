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

from packaging import version

import pytorch_lightning as pl


def has_validation_loop(trainer: pl.Trainer):
    if version.parse(pl.__version__) < version.parse("2.0.0"):
        return trainer._data_connector._val_dataloader_source.is_defined()
    return trainer.fit_loop.epoch_loop.val_loop._data_source.is_defined()
