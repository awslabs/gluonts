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

# Standard library imports
import logging
import os
from typing import Any

from gluonts.core.serde import dump_code

DEBUG = os.environ.get('DEBUG', 'false').lower() == 'true'
logging.basicConfig(level=logging.DEBUG if DEBUG else logging.INFO)

logger = logging.getLogger('SWIST')


def metric(metric: str, value: Any) -> None:
    """
    Emits a log message with a ``value`` for a specific ``metric``.

    Parameters
    ----------
    metric
        The name of the metric to be reported.
    value
        The metric value to be reported.
    """
    logger.info(f'gluonts[{metric}]: {dump_code(value)}')
