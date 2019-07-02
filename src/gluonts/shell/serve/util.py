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
import multiprocessing


def number_of_workers(settings) -> int:
    cpu_count = multiprocessing.cpu_count()

    if settings.model_server_workers:
        logging.info(
            f'Using {settings.model_server_workers} workers '
            '(set by MODEL_SERVER_WORKERS environment variable).'
        )
        return settings.model_server_workers

    elif (
        settings.sagemaker_batch
        and settings.sagemaker_max_concurrent_transforms < cpu_count
    ):
        logging.info(
            f'Using {settings.sagemaker_max_concurrent_transforms} workers '
            '(set by MaxConcurrentTransforms parameter in batch mode).'
        )
        return settings.sagemaker_max_concurrent_transforms

    else:
        logging.info(f'Using {cpu_count} workers')
        return cpu_count
