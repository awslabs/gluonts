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

import pandas as pd

from lightgbm import LGBMRegressor


class lgb_wrapper:
    """
    A wrapped of lightgbm that can be fed into the model parameters in QRX
    and TreePredictor.
    """

    def __init__(self, **lgb_params):
        self.model = LGBMRegressor(**lgb_params)

    def fit(self, train_data, train_target):
        self.model.fit(pd.DataFrame(train_data), train_target)

    def predict(self, data):
        return self.model.predict(pd.DataFrame(data))
