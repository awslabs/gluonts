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
from gluonts.dataset.artificial import constant_dataset
from gluonts.dataset.field_names import FieldName


def test_listing_1():
    """
    Test GluonTS paper examples from arxiv paper:
    https://arxiv.org/abs/1906.05264

    Listing 1
    """
    from gluonts.dataset.repository.datasets import get_dataset
    from gluonts.model.deepar import DeepAREstimator
    from gluonts.trainer import Trainer
    from gluonts.evaluation import Evaluator
    from gluonts.evaluation.backtest import backtest_metrics

    # We use electricity in the paper but that would take too long to run in
    # the unit test
    dataset_info, train_ds, test_ds = constant_dataset()

    meta = dataset_info.metadata

    estimator = DeepAREstimator(
        freq=meta.freq,
        prediction_length=1,
        trainer=Trainer(epochs=1, batch_size=32),
    )
    predictor = estimator.train(train_ds)

    evaluator = Evaluator(quantiles=(0.1, 0.5, 0.9))
    agg_metrics, item_metrics = backtest_metrics(
        train_dataset=train_ds,
        test_dataset=test_ds,
        forecaster=predictor,
        evaluator=evaluator,
    )


def test_appendix_c():
    """
    Test GluonTS paper examples from arxiv paper:
    https://arxiv.org/abs/1906.05264

    Appendix C
    """
    from typing import List
    from mxnet import gluon
    from gluonts.model.estimator import GluonEstimator
    from gluonts.model.predictor import Predictor, RepresentableBlockPredictor
    from gluonts.trainer import Trainer
    from gluonts.transform import (
        InstanceSplitter,
        Transformation,
        ExpectedNumInstanceSampler,
    )
    from gluonts.core.component import validated
    from gluonts.support.util import copy_parameters

    class MyTrainNetwork(gluon.HybridBlock):
        def __init__(self, prediction_length, cells, act_type, **kwargs):
            super().__init__(**kwargs)
            self.prediction_length = prediction_length
            with self.name_scope():
                # Set up a network that predicts the target
                self.nn = gluon.nn.HybridSequential()
                for c in cells:
                    self.nn.add(gluon.nn.Dense(units=c, activation=act_type))
                    self.nn.add(
                        gluon.nn.Dense(
                            units=self.prediction_length, activation=act_type
                        )
                    )

        def hybrid_forward(self, F, past_target, future_target):
            prediction = self.nn(past_target)
            # calculate L1 loss to learn the median
            return (prediction - future_target).abs().mean(axis=-1)

    class MyPredNetwork(MyTrainNetwork):
        # The prediction network only receives
        # past target and returns predictions
        def hybrid_forward(self, F, past_target):
            prediction = self.nn(past_target)
            return prediction.expand_dims(axis=1)

    class MyEstimator(GluonEstimator):
        @validated()
        def __init__(
            self,
            freq: str,
            prediction_length: int,
            act_type: str = "relu",
            context_length: int = 30,
            cells: List[int] = [40, 40, 40],
            trainer: Trainer = Trainer(epochs=10),
        ) -> None:
            super().__init__(trainer=trainer)
            self.freq = freq
            self.prediction_length = prediction_length
            self.act_type = act_type
            self.context_length = context_length
            self.cells = cells

        def create_training_network(self) -> MyTrainNetwork:
            return MyTrainNetwork(
                prediction_length=self.prediction_length,
                cells=self.cells,
                act_type=self.act_type,
            )

        def create_predictor(
            self,
            transformation: Transformation,
            trained_network: gluon.HybridBlock,
        ) -> Predictor:
            prediction_network = MyPredNetwork(
                prediction_length=self.prediction_length,
                cells=self.cells,
                act_type=self.act_type,
            )

            copy_parameters(trained_network, prediction_network)

            return RepresentableBlockPredictor(
                input_transform=transformation,
                prediction_net=prediction_network,
                batch_size=self.trainer.batch_size,
                freq=self.freq,
                prediction_length=self.prediction_length,
                ctx=self.trainer.ctx,
            )

        def create_transformation(self):
            # Model specific input transform
            # Here we use a transformation that randomly
            # selects training samples from all series.
            return InstanceSplitter(
                target_field=FieldName.TARGET,
                is_pad_field=FieldName.IS_PAD,
                start_field=FieldName.START,
                forecast_start_field=FieldName.FORECAST_START,
                train_sampler=ExpectedNumInstanceSampler(num_instances=1),
                past_length=self.context_length,
                future_length=self.prediction_length,
            )

    from gluonts.trainer import Trainer
    from gluonts.evaluation import Evaluator
    from gluonts.evaluation.backtest import backtest_metrics

    dataset_info, train_ds, test_ds = constant_dataset()

    meta = dataset_info.metadata
    estimator = MyEstimator(
        freq=meta.freq,
        prediction_length=1,
        trainer=Trainer(epochs=1, batch_size=32),
    )
    predictor = estimator.train(train_ds)

    evaluator = Evaluator(quantiles=(0.1, 0.5, 0.9))
    agg_metrics, item_metrics = backtest_metrics(
        train_dataset=train_ds,
        test_dataset=test_ds,
        forecaster=predictor,
        evaluator=evaluator,
    )
