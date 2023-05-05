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

from gluonts.dataset.repository.datasets import get_dataset
from gluonts.mx.trainer import Trainer

from gluonts.nursery.temporal_hierarchical_forecasting.model.cop_deepar import (
    COPDeepAREstimator,
)
from gluonts.nursery.temporal_hierarchical_forecasting.eval.evaluation import (
    evaluate_predictor,
)


def main():
    dataset = get_dataset("exchange_rate", regenerate=False)

    estimator = COPDeepAREstimator(
        freq=dataset.metadata.freq,
        prediction_length=dataset.metadata.prediction_length,
        base_estimator_name="DeepAREstimatorForCOP",
        base_estimator_hps={},
        trainer=Trainer(
            epochs=100,
            hybridize=False,
        ),
    )

    predictor = estimator.train(dataset.train)

    results = evaluate_predictor(
        predictor=predictor,
        test_dataset=dataset.test,
        evaluate_all_levels=EVALUATE_ALL_LEVELS,
        freq=dataset.metadata.freq,
    )

    print(results)


if __name__ == "__main__":
    main()
