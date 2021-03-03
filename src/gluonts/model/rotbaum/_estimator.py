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

from gluonts.core.component import validated
from gluonts.dataset.common import Dataset
from gluonts.model.estimator import Estimator, Predictor
from gluonts.transform import FilterTransformation, Transformation

from ._predictor import TreePredictor


class ThirdPartyEstimator(Estimator):
    """
    An `Estimator` that uses an external fitting mechanism, thus eliminating
    the need for a Trainer. Differs from DummyEstimator in that DummyEstimator
    does not use the training data, but merely trains at prediction time.

    Parameters
    ----------
    predictor_cls
        `Predictor` class to instantiate.
    **kwargs
        Keyword arguments to pass to the predictor constructor.
    """

    @validated()
    def __init__(self, predictor_cls: type, **kwargs) -> None:
        self.predictor = predictor_cls(**kwargs)

    def train(
        self, training_data: Dataset, validation_dataset=None
    ) -> Predictor:
        return self.predictor.train(training_data)


class TreeEstimator(ThirdPartyEstimator):
    @validated(
        getattr(TreePredictor.__init__, "Model")
    )  # Reuse the model Predictor model
    def __init__(self, **kwargs) -> None:
        super().__init__(predictor_cls=TreePredictor, **kwargs)
