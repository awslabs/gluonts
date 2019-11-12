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
import pandas as pd


class DateConstants:
    """
    Default constants for specific dates.
    """

    OLDEST_SUPPORTED_TIMESTAMP = pd.Timestamp(1800, 1, 1, 12)
    LATEST_SUPPORTED_TIMESTAMP = pd.Timestamp(2200, 1, 1, 12)
