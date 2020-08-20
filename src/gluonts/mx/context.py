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

import functools
import logging
import re
from typing import Union

import mxnet as mx

logger = logging.getLogger(__name__)


class MXContext:
    """
    Defines `custom data type validation
    <https://pydantic-docs.helpmanual.io/#custom-data-types>`_ for
    the :class:`~mxnet.context.Context` data type.
    """

    @classmethod
    def validate(cls, v: Union[str, mx.Context]) -> mx.Context:
        if isinstance(v, mx.Context):
            return v

        m = re.search(r"^(?P<dev_type>cpu|gpu)(\((?P<dev_id>\d+)\))?$", v)

        if m:
            return mx.Context(m["dev_type"], int(m["dev_id"] or 0))
        else:
            raise ValueError(
                f"bad MXNet context {v}, expected either an "
                f"mx.context.Context or its string representation"
            )

    @classmethod
    def __get_validators__(cls) -> mx.Context:
        yield cls.validate


mx.Context.validate = MXContext.validate
mx.Context.__get_validators__ = MXContext.__get_validators__


NUM_GPUS = None


def num_gpus(refresh=False):
    global NUM_GPUS
    if NUM_GPUS is None or refresh:
        n = 0
        try:
            n = mx.context.num_gpus()
        except mx.base.MXNetError as e:
            logger.error(f"Failure when querying GPU: {e}")
        NUM_GPUS = n
    return NUM_GPUS


@functools.lru_cache()
def get_mxnet_context(gpu_number=0) -> mx.Context:
    """
    Returns either CPU or GPU context
    """
    if num_gpus():
        logger.info("Using GPU")
        return mx.context.gpu(gpu_number)
    else:
        logger.info("Using CPU")
        return mx.context.cpu()


def check_gpu_support() -> bool:
    """
    Emits a log line and returns a boolean that indicate whether
    the currently installed MXNet version has GPU support.
    """
    n = num_gpus()
    logger.info(f'MXNet GPU support is {"ON" if n > 0 else "OFF"}')
    return n != 0
