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
from typing import Any
from gluonts.core.serde import dump_code

logger = logging.getLogger(__name__)


def log_metric(metric: str, value: Any) -> None:
    """
    Logs the provided metric in the format `gluonts[<metric>]: <value>`.
    """
    # pylint: disable=logging-fstring-interpolation
    logger.info(f"gluonts[{metric}]: {dump_code(value)}")
