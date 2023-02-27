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


import re
from datetime import datetime


def make_metrics(metrics_names):
    avg_epoch_loss_metric = {
        "Name": "training_loss",
        "Regex": r"'epoch_loss'=(\S+)",
    }
    final_loss_metric = {"Name": "final_loss", "Regex": r"Final loss: (\S+)"}
    other_metrics = [
        {
            "Name": metric,
            "Regex": rf"gluonts\[metric-{re.escape(metric)}\]: (\S+)",
        }
        for metric in metrics_names
    ]

    return [avg_epoch_loss_metric, final_loss_metric] + other_metrics


def make_job_name(base_job_name):
    now = datetime.utcnow().strftime("%Y-%m-%d-%H-%M-%S-%f")
    return f"{base_job_name}-{now}"
