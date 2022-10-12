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

from gluonts.ev_v5.api import evaluate
from gluonts.ev_v5.metrics import MSE, NRMSE, AbsLabel

label = np.arange(100).reshape(5, 20)
mean = label + np.random.random()

batches = [{"label": label, "mean": mean}, {"label": label, "mean": mean + 1}]

abs_label = evaluate(batches=iter(batches), metric=AbsLabel())  # non-aggregated metric
mse = evaluate(batches=iter(batches), metric=MSE(axis=1))  # aggregated metric
nrmse = evaluate(batches=iter(batches), metric=NRMSE(axis=None))  # derived metric

print(abs_label)
print(mse)
print(nrmse)


"""
[[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19]
[20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39]
[40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59]
[60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79]
[80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99]
[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19]
[20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39]
[40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59]
[60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79]
[80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99]]

[0.47393572 0.47393572 0.47393572 0.47393572 0.47393572 1.94752262
1.94752262 1.94752262 1.94752262 1.94752262]

0.031436435876320126
"""