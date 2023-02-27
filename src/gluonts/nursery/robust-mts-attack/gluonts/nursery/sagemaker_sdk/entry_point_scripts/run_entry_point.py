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


import argparse
import json
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S]",
)
logger = logging.getLogger(__name__)


def run(arguments):
    logger.info("Starting - started running custom script.")

    # TODO: WRITE YOUR CUSTOM CODE HERE

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # load hyperparameters via SM_HPS environment variable
    parser.add_argument(
        "--sm-hps", type=json.loads, default=os.environ["SM_HPS"]
    )

    # save your model here to deploy it to an endpoint later with deploy()
    parser.add_argument(
        "--model-dir", type=str, default=os.environ["SM_MODEL_DIR"]
    )
    # specified inputs (input channels) are saved here
    parser.add_argument(
        "--input-dir", type=str, default=os.environ["SM_INPUT_DIR"]
    )
    # contents of this folder will be written back to s3
    parser.add_argument(
        "--output-data-dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"]
    )

    # TODO: DONT FORGET TO PARSE ANY ADDITIONAL ARGUMENTS YOU SPECIFIED, FOR EXAMPLE THE INPUTS
    # parser.add_argument('--<your_input_var_name>', type=str, default=os.environ['SM_CHANNEL_<YOUR_INPUT_VAR_NAME>'])

    args, _ = parser.parse_known_args()

    run(args)
