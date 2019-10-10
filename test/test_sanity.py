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

# coding: utf-8

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import pytest


def test_sanity():
    # sanity test that makes sure every marker combination has at least 1 test.
    # due to https://github.com/pytest-dev/pytest/issues/812
    import gluonts as ts


@pytest.mark.gpu
def test_sanity_gpu():
    # sanity test that makes sure every marker combination has at least 1 test.
    # due to https://github.com/pytest-dev/pytest/issues/812
    import gluonts as ts


@pytest.mark.serial
def test_sanity_serial():
    # sanity test that makes sure every marker combination has at least 1 test.
    # due to https://github.com/pytest-dev/pytest/issues/812
    import gluonts as ts


@pytest.mark.gpu
@pytest.mark.serial
def test_sanity_gpu_serial():
    # sanity test that makes sure every marker combination has at least 1 test.
    # due to https://github.com/pytest-dev/pytest/issues/812
    import gluonts as ts
