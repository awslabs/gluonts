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

import logging

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S]",
)
logger = logging.getLogger(__name__)


def EMA(loss: np.ndarray, alpha: float = 0.2):
    """
    loss: one dimensional numpy array or of shape (D,1)
    alpha: value to compute the exponential moving average
    """
    loss = np.squeeze(loss)
    try:
        assert len(loss) > 0
    except AssertionError:
        logger.critical("Loss is empty, no EMA computation is done")
        return
    ema_loss = np.empty(len(loss))
    ema_loss[0] = loss[0]
    for index, item in enumerate(loss[1:]):
        ema_loss[index] = (
            alpha * loss[index] + (1 - alpha) * ema_loss[index - 1]
        )

    return ema_loss
