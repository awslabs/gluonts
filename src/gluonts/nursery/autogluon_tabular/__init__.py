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

from gluonts.model.predictor import Localizer

from .estimator import TabularEstimator


def LocalTabularPredictor(*args, **kwargs) -> Localizer:
    """
    A predictor that trains an ad-hoc model for each time series that it is
    given to predict.

    The constructor arguments are the same as for ``TabularEstimator``.
    """
    return Localizer(TabularEstimator(*args, **kwargs))


__all__ = ["TabularEstimator", "LocalTabularPredictor"]

# fix Sphinx issues, see https://bit.ly/2K2eptM
for item in __all__:
    if hasattr(item, "__module__"):
        setattr(item, "__module__", __name__)
