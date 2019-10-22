# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import argparse
import os
import json
import time


def run(arguments):
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), "Starting - started running custom script.")
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # load hyperparameters via SM_HPS environment variable
    parser.add_argument('--sm_hps', type=json.loads, default=os.environ['SM_HPS'])

    # input data, output dir and model directories
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--input_dir', type=str, default=os.environ['SM_INPUT_DIR'])
    # contents will be written back to s3
    parser.add_argument('--output_data_dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])

    args, _ = parser.parse_known_args()

    run(args)
