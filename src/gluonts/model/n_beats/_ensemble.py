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

# Standard library imports
from itertools import product
from pathlib import Path
from typing import List, Optional, Iterator, Callable
import copy
import os
import logging

# Third-party imports
import mxnet as mx
import numpy as np
from pydantic import ValidationError

# First-party imports
from gluonts.core.component import validated, from_hyperparameters
from gluonts.core import fqname_for
from gluonts.core.serde import dump_json, load_json
from gluonts.dataset.common import Dataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.loader import DataBatch
from gluonts.model.estimator import Estimator
from gluonts.model.forecast import Forecast, SampleForecast
from gluonts.model.predictor import Predictor, RepresentableBlockPredictor
from gluonts.core.exception import GluonTSHyperparametersError
from gluonts.trainer import Trainer

# Relative imports
from ._network import VALID_LOSS_FUNCTIONS
from ._estimator import NBEATSEstimator

# None is also a valid parameter
AGGREGATION_METHODS = "median", "mean", "none"


class NBEATSEnsemblePredictor(Predictor):
    """"
    An ensemble predictor for N-BEATS.
    Calling '.predict' will result in::

        |predictors|x|dataset|

    predictions, if aggregation_method is 'none', otherwise in::

        |dataset|

    Parameters
    ----------
    prediction_length
        Prediction horizon.
    freq
        Frequency of the predicted data.
    predictors
        The list of 'RepresentableBlockPredictor' that the ensemble consists of.
    aggregation_method
        The method by which to aggregate the individual predictions of the models.
        Either 'median', 'mean' or 'none', in which case no aggregation happens.
        Default is 'median'.
    """

    def __init__(
        self,
        prediction_length: int,
        freq: str,
        predictors: List[RepresentableBlockPredictor],
        aggregation_method: Optional[str] = "median",
    ) -> None:
        super().__init__(freq=freq, prediction_length=prediction_length)

        assert aggregation_method in AGGREGATION_METHODS

        self.predictors = predictors
        self.aggregation_method = aggregation_method

    def set_aggregation_method(self, aggregation_method: str):
        assert aggregation_method in AGGREGATION_METHODS
        self.aggregation_method = aggregation_method

    def serialize(self, path: Path) -> None:
        # serialize some metadata
        super().serialize(path)

        # basically save each predictor in its own sub-folder
        num_digits = len(str(len(self.predictors)))
        for index, predictor in enumerate(self.predictors):
            composite_path = path / f"predictor_{str(index).zfill(num_digits)}"
            os.makedirs(str(composite_path))
            predictor.serialize(composite_path)

        # serialize all remaining constructor parameters
        with (path / "parameters.json").open("w") as fp:
            parameters = dict(
                prediction_length=self.prediction_length,
                freq=self.freq,
                aggregation_method=self.aggregation_method,
                num_predictors=len(self.predictors),
            )
            print(dump_json(parameters), file=fp)

    @classmethod
    def deserialize(
        cls, path: Path, ctx: Optional[mx.Context] = None
    ) -> "NBEATSEnsemblePredictor":
        # deserialize constructor parameters
        with (path / "parameters.json").open("r") as fp:
            parameters = load_json(fp.read())

        # basically save each predictor in its own sub-folder
        num_predictors = parameters["num_predictors"]
        num_digits = len(str(num_predictors))

        predictors = []
        # load all the predictors individually and also make sure not to load anything else by mistake
        predictor_locations = [
            f"predictor_{str(index).zfill(num_digits)}"
            for index in range(num_predictors)
        ]

        # deserialize predictors
        for sub_dir in predictor_locations:
            predictors.append(
                RepresentableBlockPredictor.deserialize(path / sub_dir, ctx)
            )

        return NBEATSEnsemblePredictor(
            prediction_length=parameters["prediction_length"],
            freq=parameters["freq"],
            predictors=predictors,
            aggregation_method=parameters["aggregation_method"],
        )

    # Additionally implemented since we are dealing with RepresentableBlockPredictor
    def hybridize(self, batch: DataBatch) -> None:
        for predictor in self.predictors:
            predictor.hybridize(batch)

    def predict(
        self, dataset: Dataset, num_samples: Optional[int] = 1, **kwargs
    ) -> Iterator[Forecast]:
        if num_samples != 1:
            logging.warning(
                "NBEATSEnsemblePredictor does not support sampling. "
                "Therefore 'num_samples' will be ignored and set to 1."
            )
        iterators = []

        # create the iterators from the predictors
        for predictor in self.predictors:
            iterators.append(predictor.predict(dataset, num_samples=1))

        # we always have to predict for one series in the dataset with
        # all models and return it as a 'SampleForecast' so that it is
        # clear that all these prediction concern the same series
        for item in dataset:
            output = []
            start_date = None

            for iterator in iterators:
                prediction = next(iterator)

                # on order to avoid mypys complaints
                assert isinstance(prediction, SampleForecast)

                output.append(prediction.samples)

                # get the forecast start date
                if start_date is None:
                    start_date = prediction.start_date
            output = np.stack(output, axis=0)

            # aggregating output of different models
            # default according to paper is median,
            # but we can also make use of not aggregating
            if self.aggregation_method == "median":
                output = np.median(output, axis=0)
            elif self.aggregation_method == "mean":
                output = np.mean(output, axis=0)
            else:  # "none": do not aggregate
                pass

            # on order to avoid mypys complaints
            assert start_date is not None

            yield SampleForecast(
                output,
                start_date=start_date,
                freq=start_date.freqstr,
                item_id=item[FieldName.ITEM_ID]
                if FieldName.ITEM_ID in item
                else None,
                info=item["info"] if "info" in item else None,
            )

    def __eq__(self, that):
        """
        Unfortunately it cannot be guaranteed that two predictors are not equal if this returns false
        if for some reason the order of the predictors list has been altered.
        """
        if type(self) != type(that):
            return False

        try:
            if not (
                self.prediction_length == that.prediction_length
                and self.freq == that.freq
                and self.aggregation_method == that.aggregation_method
            ):
                return False
        except ValueError:
            # For example if the order of the predictors is changed we get this error
            return False

        for own_predictor, that_predictor in zip(
            self.predictors, that.predictors
        ):
            if not own_predictor == that_predictor:
                return False

        return True


class NBEATSEnsembleEstimator(Estimator):
    """
    An ensemble N-BEATS Estimator (approximately) as described
    in the paper:  https://arxiv.org/abs/1905.10437.

    The three meta parameters 'meta_context_length', 'meta_loss_function' and 'meta_bagging_size'
    together define the way the sub-models are assembled together.
    The total number of models used for the ensemble is::

        |meta_context_length| x |meta_loss_function| x meta_bagging_size

    Noteworthy differences in this implementation compared to the paper:
    * The parameter L_H is not implemented; we sample training sequences
    using the default method in GluonTS using the "InstanceSplitter".

    Parameters
    ----------
    freq
        Time time granularity of the data
    prediction_length
        Length of the prediction. Also known as 'horizon'.
    meta_context_length
        The different 'context_length' (aslso known as 'lookback period')
        to use for training the models.
        The 'context_length' is the number of time units that condition the predictions.
        Default and recommended value: [multiplier * prediction_length for multiplier in range(2, 7)]
    meta_loss_function
        The different 'loss_function' (also known as metric) to use for training the models.
        Unlike other models in GluonTS this network does not use a distribution.
        Default and recommended value: ["sMAPE", "MASE", "MAPE"]
    meta_bagging_size
        The number of models that share the parameter combination of 'context_length'
        and 'loss_function'. Each of these models gets a different initialization random initialization.
        Default and recommended value: 10
    trainer
        Trainer object to be used (default: Trainer())
    num_stacks:
        The number of stacks the network should contain.
        Default and recommended value for generic mode: 30
        Recommended value for interpretable mode: 2
    num_blocks
        The number of blocks per stack.
        A list of ints of length 1 or 'num_stacks'.
        Default and recommended value for generic mode: [1]
        Recommended value for interpretable mode: [3]
    block_layers
        Number of fully connected layers with ReLu activation per block.
        A list of ints of length 1 or 'num_stacks'.
        Default and recommended value for generic mode: [4]
        Recommended value for interpretable mode: [4]
    widths
        Widths of the fully connected layers with ReLu activation in the blocks.
        A list of ints of length 1 or 'num_stacks'.
        Default and recommended value for generic mode: [512]
        Recommended value for interpretable mode: [256, 2048]
    sharing
        Whether the weights are shared with the other blocks per stack.
        A list of ints of length 1 or 'num_stacks'.
        Default and recommended value for generic mode: [False]
        Recommended value for interpretable mode: [True]
    expansion_coefficient_lengths
        If the type is "G" (generic), then the length of the expansion coefficient.
        If type is "T" (trend), then it corresponds to the degree of the polynomial.
        If the type is "S" (seasonal) then its not used.
        A list of ints of length 1 or 'num_stacks'.
        Default value for generic mode: [32]
        Recommended value for interpretable mode: [3]
    stack_types
        One of the following values: "G" (generic), "S" (seasonal) or "T" (trend).
        A list of strings of length 1 or 'num_stacks'.
        Default and recommended value for generic mode: ["G"]
        Recommended value for interpretable mode: ["T","S"]
    **kwargs
        Arguments passed down to the individual estimators.
    """

    # The validated() decorator makes sure that parameters are checked by
    # Pydantic and allows to serialize/print models. Note that all parameters
    # have defaults except for `freq` and `prediction_length`. which is
    # recommended in GluonTS to allow to compare models easily.
    @validated()
    def __init__(
        self,
        freq: str,
        prediction_length: int,
        meta_context_length: Optional[List[int]] = None,
        meta_loss_function: Optional[List[str]] = None,
        meta_bagging_size: int = 10,
        trainer: Trainer = Trainer(),
        num_stacks: int = 30,
        widths: Optional[List[int]] = None,
        num_blocks: Optional[List[int]] = None,
        num_block_layers: Optional[List[int]] = None,
        expansion_coefficient_lengths: Optional[List[int]] = None,
        sharing: Optional[List[bool]] = None,
        stack_types: Optional[List[str]] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        assert (
            prediction_length > 0
        ), "The value of `prediction_length` should be > 0"

        self.freq = freq
        self.prediction_length = prediction_length

        assert meta_loss_function is None or all(
            [
                loss_function in VALID_LOSS_FUNCTIONS
                for loss_function in meta_loss_function
            ]
        ), f"Each loss function has to be one of the following: {VALID_LOSS_FUNCTIONS}."
        assert meta_context_length is None or all(
            [context_length > 0 for context_length in meta_context_length]
        ), "The value of each `context_length` should be > 0"
        assert (
            meta_bagging_size is None or meta_bagging_size > 0
        ), "The value of each `context_length` should be > 0"

        self.meta_context_length = (
            meta_context_length
            if meta_context_length is not None
            else [multiplier * prediction_length for multiplier in range(2, 8)]
        )
        self.meta_loss_function = (
            meta_loss_function
            if meta_loss_function is not None
            else VALID_LOSS_FUNCTIONS
        )
        self.meta_bagging_size = meta_bagging_size

        # The following arguments are validated in the NBEATSEstimator:
        self.trainer = trainer
        print(f"TRAINER:{str(trainer)}")
        self.num_stacks = num_stacks
        self.widths = widths
        self.num_blocks = num_blocks
        self.num_block_layers = num_block_layers
        self.expansion_coefficient_lengths = expansion_coefficient_lengths
        self.sharing = sharing
        self.stack_types = stack_types

        # Actually instantiate the different models
        self.estimators = self._estimator_factory(**kwargs)

    def _estimator_factory(self, **kwargs):
        estimators = []
        for context_length, loss_function, init_id in product(
            self.meta_context_length,
            self.meta_loss_function,
            list(range(self.meta_bagging_size)),
        ):
            # So far no use for the init_id, models are by default always randomly initialized
            estimators.append(
                NBEATSEstimator(
                    freq=self.freq,
                    prediction_length=self.prediction_length,
                    context_length=context_length,
                    trainer=copy.deepcopy(self.trainer),
                    num_stacks=self.num_stacks,
                    widths=self.widths,
                    num_blocks=self.num_blocks,
                    num_block_layers=self.num_block_layers,
                    expansion_coefficient_lengths=self.expansion_coefficient_lengths,
                    sharing=self.sharing,
                    stack_types=self.stack_types,
                    loss_function=loss_function,
                    **kwargs,
                )
            )
        return estimators

    @classmethod
    def from_hyperparameters(
        cls, **hyperparameters
    ) -> "NBEATSEnsembleEstimator":
        Model = getattr(cls.__init__, "Model", None)

        if not Model:
            raise AttributeError(
                f"Cannot find attribute Model attached to the "
                f"{fqname_for(cls)}. Most probably you have forgotten to mark "
                f"the class constructor as @validated()."
            )

        try:
            trainer = from_hyperparameters(Trainer, **hyperparameters)
            return cls(
                **Model(**{**hyperparameters, "trainer": trainer}).__dict__
            )
        except ValidationError as e:
            raise GluonTSHyperparametersError from e

    def train(
        self, training_data: Dataset, validation_data: Optional[Dataset] = None
    ) -> NBEATSEnsemblePredictor:
        predictors = []

        for index, estimator in enumerate(self.estimators):
            logging.info(
                f"Training estimator {index + 1}/{len(self.estimators)}."
            )
            predictors.append(estimator.train(training_data, validation_data))

        return NBEATSEnsemblePredictor(
            self.prediction_length, self.freq, predictors
        )
