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

# First-party imports
from gluonts.dataset.common import TimeSeriesItem
from gluonts.dataset.jsonl import JsonLinesFile


def test_file(path):
    for raw_dataset in JsonLinesFile(path):
        TimeSeriesItem.parse_obj(raw_dataset.content)
    print('ok')


def run():
    import sys

    test_file(sys.argv[1])


if __name__ == '__main__':
    run()
