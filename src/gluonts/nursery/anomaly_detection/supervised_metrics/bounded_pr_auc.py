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

import numpy as np
from sklearn.metrics import auc


def bounded_pr_auc(
    precisions: np.array, recalls: np.array, lower_bound: float = 0
) -> float:
    """Bounded PR AUC --> AUC when recall > lower_bound

    Parameters
    ----------
    precisions : np.array
        precisions of different thresholds
    recalls : np.array
        recalls of different thresholds
    lower_bound : float
        lower bound of recalls

    Returns
    -------
    bounded PR-AUC : float
    """
    sorted_recalls, sorted_precisions = zip(
        *sorted(zip(recalls, precisions), key=lambda x: (x[0], x[1]))
    )
    arg_num = np.argmax(np.array(sorted_recalls) >= lower_bound)
    pr_auc = auc(sorted_recalls[arg_num:], sorted_precisions[arg_num:])
    return pr_auc
