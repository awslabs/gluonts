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
from gluonts.core.component import validated
from gluonts.model.estimator import DummyEstimator

# Relative imports
from ._predictor import Naive2Predictor


class Naive2Estimator(DummyEstimator):
    """
    An estimator that, upon `train`, simply returns a pre-constructed `Naive2Predictor`.

    Parameters
    ----------
    kwargs
        Arguments to pass to the `Naive2Predictor` constructor.
    """

    @validated(
        getattr(Naive2Predictor.__init__, "Model")
    )  # Reuse the model Predictor model
    def __init__(self, **kwargs) -> None:
        super().__init__(predictor_cls=Naive2Predictor, **kwargs)
