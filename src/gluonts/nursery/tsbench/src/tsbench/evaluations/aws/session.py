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

import boto3

_SESSION = None


def default_session() -> boto3.Session:
    """
    Returns the shared session to be used.
    """
    global _SESSION  # pylint: disable=global-statement
    if _SESSION is None:
        _SESSION = boto3.Session()  # type: ignore
    return _SESSION


def account_id() -> boto3.Session:
    """
    Returns the ID of the account.
    """
    return default_session().client("sts").get_caller_identity().get("Account")
