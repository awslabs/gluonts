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

from gluonts.ev.metrics import MSE, NRMSE

label = np.arange(100).reshape(5, 20)
mean = label + np.random.random()

batches = [{"label": label, "mean": mean}, {"label": label, "mean": mean + 1}]

mse = MSE(axis=1).evaluate_batches(batches=iter(batches))
nrmse = NRMSE().evaluate_batches(batches=iter(batches))

print(mse)
print(nrmse)
