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

import itertools
import pickle
from functools import partial
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union, cast

import mxnet as mx
import numpy as np
import pandas as pd
from mxnet.gluon import HybridBlock

from gluonts.core.component import DType, validated
from gluonts.dataset.common import DataEntry, Dataset, ListDataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.loader import (
    InferenceDataLoader,
    TrainDataLoader,
    ValidationDataLoader,
)
from gluonts.dataset.util import to_pandas
from gluonts.itertools import Cached
from gluonts.model.estimator import Estimator
from gluonts.model.predictor import Predictor
from gluonts.mx.batchify import batchify
from gluonts.mx.distribution import DistributionOutput
from gluonts.mx.model.estimator import TrainOutput
from gluonts.mx.model.predictor import RepresentableBlockPredictor
from gluonts.mx.trainer import Trainer
from gluonts.mx.trainer.model_iteration_averaging import (
    IterationAveragingStrategy,
)
from gluonts.mx.util import copy_parameters, get_hybrid_forward_input_names
from gluonts.time_feature import get_lags_for_frequency
from gluonts.transform import (
    AddObservedValuesIndicator,
    AddTimeFeatures,
    AsNumpyArray,
    Chain,
    DummyValueImputation,
    ExpectedNumInstanceSampler,
    MapTransformation,
    MissingValueImputation,
    RemoveFields,
    RenameFields,
    SelectFields,
    SetField,
    SimpleTransformation,
    SwapAxes,
    Transformation,
    TransformedDataset,
    VstackFeatures,
)

from ..forecast_generator import StatefulDistributionForecastGenerator
from ..masking import SimpleMaskingStrategy
from ..native import ema
from ..predictor import NETWORK_STATE_KEY
from ..transform import (
    AddShiftedTimestamp,
    AnomalyScoringSplitter,
    LeadtimeShifter,
)
from ._defaults import RnnDefaults
from ._network import StreamingRnnPredictNetwork, StreamingRnnTrainNetwork

LAGS_STATE_KEY = "s:lag"
SCALE_STATE_KEY = "s:scale"
FSC_STATE_KEY = "s:feat_static_cat"

SUPPORTED_FREQS = ["1min", "5min", "10min", "1H", "1D"]

DEFAULT_LAGS_UB = {
    "1min": 1 * 7 * 24 * 60 + 3 * 60,  # one week + 3 hours,
    "5min": 2 * 7 * 24 * 12 + 3 * 12,  # two weeks + 3 hours
    "10min": 2 * 7 * 24 * 6 + 3 * 6,  # two weeks + 3 hours
    "1H": 4 * 7 * 24 + 3,  # four weeks + 3 hours
    "1D": 2 * 365,  # two year
}

DEFAULT_SMALL_LAGS = {
    "1min": list(range(1, 61)),
    "5min": list(range(1, 37)),
    "10min": list(range(1, 25)),
    "1H": list(range(1, 25)),
    "1D": list(range(1, 22)),
}


class AddStreamScale(SimpleTransformation):
    @validated()
    def __init__(
        self,
        target_field: str,
        scale_field: str,
        state_field: str,
        minimum_value: float,
        initial_scale: float,
        alpha: float,
    ) -> None:
        self.target_field = target_field
        self.scale_field = scale_field
        self.state_field = state_field
        self.minimum_value = minimum_value
        self.initial_scale = initial_scale
        self.alpha = alpha

    def transform(self, data: DataEntry) -> DataEntry:
        target = data[self.target_field]
        scale_state = data.get(self.state_field)

        scale, new_scale_state = ema(
            target,
            alpha=self.alpha,
            minimum_value=self.minimum_value,
            initial_scale=self.initial_scale,
            state=scale_state,
        )

        data[self.scale_field] = scale
        data[self.state_field] = np.array(new_scale_state)
        return data


class AddLags(SimpleTransformation):
    @validated()
    def __init__(
        self,
        lag_seq: List[int],
        lag_field: str,
        target_field: str,
        lag_state_field: str,
    ) -> None:
        self.lag_seq = sorted(lag_seq)
        self.lag_field = lag_field
        self.target_field = target_field
        self.lag_state_field = lag_state_field
        self.max_lag = self.lag_seq[-1]

    def transform(self, data: DataEntry) -> DataEntry:
        target = data[self.target_field]
        buffer = data.get(self.lag_state_field)
        if buffer is None:
            t = np.concatenate([np.zeros(self.max_lag), target])
        else:
            t = np.concatenate([buffer, target])
        lags = np.vstack(
            [t[self.max_lag - l : len(t) - l] for l in self.lag_seq]
        )
        data[self.lag_field] = np.nan_to_num(lags)
        data[self.lag_state_field] = t[-self.max_lag :]
        return data


class AddStreamAggregateLags(SimpleTransformation):
    @validated()
    def __init__(
        self,
        target_field: str,
        output_field: str,
        lag_state_field: str,
        lead_time: int,
        base_freq: str,
        agg_freq: str,
        agg_lags: List[int],
        agg_fun: str = "mean",
        dtype: DType = np.float32,
    ) -> None:

        self.target_field = target_field
        self.feature_name = output_field
        self.lead_time = lead_time
        self.base_freq = base_freq
        self.agg_freq = agg_freq
        self.agg_lags = agg_lags
        self.agg_fun = agg_fun
        self.lag_state_field = lag_state_field
        self.dtype = dtype

        self.ratio = pd.Timedelta(self.agg_freq) / pd.Timedelta(self.base_freq)
        assert (
            self.ratio.is_integer() and self.ratio >= 1
        ), "The aggregate frequency should be a multiple of the base frequency."
        self.ratio = int(self.ratio)

        # convert lags to original freq and adjust based on lead time
        adj_lags = [
            lag * self.ratio - (self.lead_time - 1) for lag in self.agg_lags
        ]

        self.half_window = (self.ratio - 1) // 2
        valid_adj_lags = [x for x in adj_lags if x - self.half_window > 0]
        self.valid_lags = [
            int(np.ceil(x / self.ratio)) for x in valid_adj_lags
        ]
        self.offset = (self.lead_time - 1) % self.ratio

        assert len(self.valid_lags) > 0

        if len(self.agg_lags) - len(self.valid_lags) > 0:
            print(
                f"The aggregate lags {set(self.agg_lags[:- len(self.valid_lags)])} "
                f"of frequency {self.agg_freq} are ignored."
            )

        self.max_state_lag = max(self.valid_lags)

    def transform(self, data: DataEntry) -> DataEntry:
        assert self.base_freq == data["start"].freq

        buffer = data.get(self.lag_state_field)
        if buffer is None:
            t = data[self.target_field]
            t_agg = (pd.Series(t).rolling(self.ratio).agg(self.agg_fun))[
                self.ratio - 1 :
            ]
        else:
            t = np.concatenate(
                [buffer["base_target"], data[self.target_field]]
            )
            new_agg_lags = (
                pd.Series(t).rolling(self.ratio).agg(self.agg_fun)
            )[self.ratio - 1 :]
            t_agg = pd.Series(
                np.concatenate([buffer["agg_lags"], new_agg_lags.values])
            )

        # compute the aggregate lags for each time point of the time series
        agg_vals = np.concatenate(
            [
                np.zeros(
                    (max(self.valid_lags) * self.ratio + self.half_window,)
                ),
                t_agg.values,
            ],
            axis=0,
        )
        lags = np.vstack(
            [
                agg_vals[
                    -(
                        l * self.ratio
                        - self.offset
                        - self.half_window
                        + len(data[self.target_field])
                        - 1
                    ) : -(l * self.ratio - self.offset - self.half_window - 1)
                    if -(l * self.ratio - self.offset - self.half_window - 1)
                    is not 0
                    else None
                ]
                for l in self.valid_lags
            ]
        )

        # update the data entry
        data[self.feature_name] = np.nan_to_num(lags)
        data[self.lag_state_field] = {
            "agg_lags": t_agg.values[-self.max_state_lag * self.ratio + 1 :],
            "base_target": t[-self.ratio + 1 :] if self.ratio > 1 else [],
        }

        assert data[self.feature_name].shape == (
            len(self.valid_lags),
            len(data[self.target_field]),
        )

        return data


class CopyField(SimpleTransformation):
    """
    Copies the value of input_field into output_field and does nothing
    if input_field is not present or None.
    """

    @validated()
    def __init__(
        self,
        output_field: str,
        input_field: str,
    ) -> None:
        self.output_field = output_field
        self.input_field = input_field

    def transform(self, data: DataEntry) -> DataEntry:

        field = data.get(self.input_field)
        if field is not None:
            data[self.output_field] = data[self.input_field].copy()

        return data


class StateInitializer:
    @validated()
    def __init__(self, pickled_state: str) -> None:
        self.pickled_bytes = pickled_state.encode("latin1")

    def __call__(self) -> Any:
        return pickle.loads(self.pickled_bytes)

    @staticmethod
    def from_state(state) -> "StateInitializer":
        s = pickle.dumps(state).decode("latin1")
        return StateInitializer(s)


class StreamingRnnEstimator(Estimator):
    @validated()
    def __init__(
        self,
        freq: str = RnnDefaults.FREQ,
        lead_time: int = RnnDefaults.LEAD_TIME,
        train_window_length: int = RnnDefaults.TRAIN_WINDOW_LENGTH,
        skip_initial_window_pct: float = RnnDefaults.SKIP_INITIAL_WINDOW_PCT,
        hidden_size: int = RnnDefaults.HIDDEN_SIZE,
        num_layers: int = RnnDefaults.NUM_LAYERS,
        trainer: Trainer = Trainer(**RnnDefaults.TRAINER_KWARGS),
        distr_output: DistributionOutput = RnnDefaults.DISTR_OUTPUT,
        use_feat_static_cat: bool = RnnDefaults.USE_FEAT_STATIC_CAT,
        cardinality: Optional[List[int]] = RnnDefaults.CARDINALITY,
        embedding_dimension: Optional[
            List[int]
        ] = RnnDefaults.EMBEDDING_DIMENSION,
        dropout_type: Optional[str] = RnnDefaults.DROPOUT_TYPE,
        dropout_rate: float = RnnDefaults.DROPOUT_RATE,
        lags: Optional[List[Tuple[str, List[int], str]]] = RnnDefaults.LAGS,
        imputation: Optional[
            Union[str, MissingValueImputation]
        ] = RnnDefaults.IMPUTATION,
        alpha: float = RnnDefaults.ALPHA,
        beta: float = RnnDefaults.BETA,
        ransac_thresh: Optional[List[float]] = RnnDefaults.RANSAC_THRESH,
        hybridize_prediction_net: bool = RnnDefaults.HYBRIDIZE_PRED_NET,
        batch_size: int = RnnDefaults.BATCH_SIZE,
        cache_data: bool = False,
        dtype: DType = np.float32,
    ) -> None:
        super().__init__(lead_time=lead_time)

        assert lead_time > 0
        assert train_window_length > 0
        assert 0 <= skip_initial_window_pct < 1
        assert hidden_size > 0
        assert num_layers > 0
        assert alpha >= 0
        assert beta >= 0
        assert batch_size > 0

        self.hybridize_prediction_net = hybridize_prediction_net
        self.cache_data = cache_data
        self.freq = freq
        self.train_window_length = train_window_length
        self.skip_initial_window = min(
            int(train_window_length * skip_initial_window_pct),
            train_window_length - 1,
        )
        self.trainer = trainer
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.distr_output = distr_output
        self.dropout_type = dropout_type
        self.dropout_rate = dropout_rate
        self.imputation = imputation
        self.alpha = alpha
        self.beta = beta
        self.batch_size = batch_size
        self.dtype = dtype

        assert (cardinality and use_feat_static_cat) or (
            not (cardinality or use_feat_static_cat)
        ), "You should set `cardinality` if and only if `use_feat_static_cat=True`"
        assert cardinality is None or all(
            [c > 0 for c in cardinality]
        ), "Elements of `cardinality` should be > 0"
        assert embedding_dimension is None or all(
            [e > 0 for e in embedding_dimension]
        ), "Elements of `embedding_dimension` should be > 0"
        self.use_feat_static_cat = use_feat_static_cat
        self.cardinality = (
            cardinality if cardinality and use_feat_static_cat else [1]
        )
        self.embedding_dimension = (
            embedding_dimension
            if embedding_dimension is not None
            else [min(20, (cat + 1) // 2) for cat in self.cardinality]
        )

        if ransac_thresh:
            assert isinstance(ransac_thresh, list)
            assert all(th > 0 for th in ransac_thresh)
            self.ransac_thresh = ransac_thresh
        else:
            self.ransac_thresh = []

        base_lag_tf = AddStreamAggregateLags(
            target_field="target",
            output_field="base_lags",
            lag_state_field=LAGS_STATE_KEY,
            lead_time=self.lead_time,
            base_freq=self.freq,
            agg_freq=self.freq,
            agg_lags=self._get_default_lags_for_frequency(),
        )

        agg_lags_tf = []
        if lags:
            for agg_freq, agg_lags, agg_fun in lags:
                agg_lags_tf.append(
                    AddStreamAggregateLags(
                        target_field="target",
                        output_field=f"agg_lags_{agg_freq}",
                        lag_state_field=f"{LAGS_STATE_KEY}_{agg_freq}",
                        lead_time=self.lead_time,
                        base_freq=self.freq,
                        agg_freq=agg_freq,
                        agg_lags=agg_lags,
                        agg_fun=agg_fun,
                    )
                )

        self.lags = [base_lag_tf] + agg_lags_tf

    def _get_default_lags_for_frequency(self) -> List[int]:
        # Lags are not processed here based on the lead_time.
        if self.freq in SUPPORTED_FREQS:
            lags_from_freq = get_lags_for_frequency(
                freq_str=self.freq, lag_ub=DEFAULT_LAGS_UB[self.freq]
            )

            lags = sorted(set(DEFAULT_SMALL_LAGS[self.freq] + lags_from_freq))
        else:
            lags = get_lags_for_frequency(freq_str=self.freq)

        # Lags are defined as the number of steps before a target timestamp,
        # i.e., for predicting the value at time t we can use the values t-l where l > 0.
        # Therefore, lags are positive values.
        assert min(lags) > 0
        return lags

    def _data_transformation_steps(self) -> List:
        if self.use_feat_static_cat:
            # In training the data has the "feat_static_cat" field for hybrid_forward. The state is irrelevant.
            # In inference the data has the state field "s:feat_static_cat" and the "feat_static_cat" field
            # for hybrid_forward needs to be created.
            fsc_field = [
                CopyField(
                    output_field=FieldName.FEAT_STATIC_CAT,
                    input_field=FSC_STATE_KEY,
                )
            ]
        else:
            fsc_field = [
                SetField(output_field=FieldName.FEAT_STATIC_CAT, value=[0.0])
            ]

        tf_chain = [
            RemoveFields(field_names=["anomaly_indicator"]),
            AsNumpyArray(field=FieldName.TARGET, expected_ndim=1),
            AsNumpyArray(
                field=FieldName.FEAT_STATIC_CAT,
                expected_ndim=1,
                dtype=self.dtype,
            ),
            AddStreamScale(
                target_field="target",
                scale_field="scale",
                state_field=SCALE_STATE_KEY,
                minimum_value=1.0e-10,
                initial_scale=1.0e-10,
                alpha=0.002,
            ),
            AddObservedValuesIndicator(
                target_field=FieldName.TARGET,
                output_field=FieldName.OBSERVED_VALUES,
                imputation_method=self.imputation,
            ),
            VstackFeatures(
                output_field=FieldName.FEAT_TIME,
                input_fields=[FieldName.OBSERVED_VALUES],
                drop_inputs=False,
            ),
        ]

        lag_stack = [
            VstackFeatures(
                output_field="lags",
                input_fields=[l.feature_name for l in self.lags],
                drop_inputs=True,
            ),
        ]
        return fsc_field + tf_chain + self.lags + lag_stack

    def _training_batch_maker(self) -> List:
        return [
            LeadtimeShifter(
                lead_time=self.lead_time,
                target_field=FieldName.TARGET,
                time_series_fields=[
                    FieldName.FEAT_TIME,
                    FieldName.OBSERVED_VALUES,
                ],
            ),
            AnomalyScoringSplitter(
                target_field="label_target",
                is_pad_field=FieldName.IS_PAD,
                start_field=FieldName.START,
                forecast_start_field=FieldName.FORECAST_START,
                train_sampler=ExpectedNumInstanceSampler(num_instances=1),
                train_window_length=self.train_window_length,
                output_NTC=True,
                pick_incomplete=False,
                time_series_fields=[
                    "input_target",
                    FieldName.FEAT_TIME,
                    FieldName.OBSERVED_VALUES,
                    "lags",
                    "scale",
                ],
            ),
        ]

    def _inference_batch_maker(self) -> List:
        return [
            RenameFields({"target": "input_target"}),
            SwapAxes(input_fields=[FieldName.FEAT_TIME, "lags"], axes=(0, 1)),
            AddShiftedTimestamp(
                input_field="start",
                output_field="forecast_start",
                shift=self.lead_time + 1,
            ),
        ]

    def create_transformation(self) -> Chain:
        return Chain(
            self._data_transformation_steps() + self._training_batch_maker()
        )

    # defines the network, we get to see one batch to initialize it.
    # the network should return at least one tensor that is used as a loss to minimize in the training loop.
    # several tensors can be returned for instance for analysis, see DeepARTrainingNetwork for an example.
    def create_training_network(self) -> HybridBlock:
        return StreamingRnnTrainNetwork(
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            distr_output=self.distr_output,
            cardinality=self.cardinality,
            embedding_dimension=self.embedding_dimension,
            dropout_type=self.dropout_type,
            dropout_rate=self.dropout_rate,
            skip_initial_window=self.skip_initial_window,
            alpha=self.alpha,
            beta=self.beta,
        )

    # we now define how the prediction happens given that we are provided a
    # training network.
    def create_predictor(
        self, transformation: Transformation, trained_network: HybridBlock
    ) -> RepresentableBlockPredictor:
        prediction_network = StreamingRnnPredictNetwork(
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            distr_output=self.distr_output,
            cardinality=self.cardinality,
            embedding_dimension=self.embedding_dimension,
            dropout_type=self.dropout_type,
            dropout_rate=self.dropout_rate,
            skip_initial_window=self.skip_initial_window,
        )

        copy_parameters(
            net_source=trained_network, net_dest=prediction_network
        )

        inference_transformation = Chain(
            self._data_transformation_steps() + self._inference_batch_maker()
        )

        return RepresentableBlockPredictor(
            input_transform=inference_transformation,
            forecast_generator=StatefulDistributionForecastGenerator(
                self.distr_output
            ),
            prediction_net=prediction_network,
            batch_size=1,
            freq=self.freq,
            prediction_length=1,
            lead_time=self.lead_time,
            ctx=self.trainer.ctx,
        )

    def train(
        self,
        training_data: Dataset,
        validation_data: Optional[Dataset] = None,
        num_workers: Optional[int] = None,
    ) -> Predictor:
        assert isinstance(training_data, ListDataset)

        train_output = self._ransac(
            training_data, validation_data, num_workers
        )

        predictor = cast(RepresentableBlockPredictor, train_output.predictor)

        if self.hybridize_prediction_net:
            state_init = self.get_state_initializer()

            # inputs = [batch[k] for k in input_names]

            one_sample = list(itertools.islice(training_data, 1))[0].copy()
            one_sample.update(**state_init())
            inf_loader = InferenceDataLoader(
                dataset=[one_sample],
                stack_fn=partial(
                    batchify, ctx=self.trainer.ctx, dtype=self.dtype
                ),
                batch_size=predictor.batch_size,
                transform=predictor.input_transform,
            )
            d = list(inf_loader)[0]
            return predictor.as_symbol_block_predictor(d)
        else:
            return predictor

    def _ransac(
        self,
        training_data: ListDataset,
        validation_data: Optional[Dataset] = None,
        num_workers: Optional[int] = None,
    ) -> TrainOutput:
        num_ransac_passes = len(self.ransac_thresh)
        cur_ransac_pass = 0
        train_model = True
        while cur_ransac_pass < num_ransac_passes:
            if train_model:
                print("RANSAC: Model training.")
                train_output = self.train_model(
                    training_data, validation_data, num_workers=num_workers
                )
                train_model = False
            else:
                print("RANSAC: Skip model training.")

            print(
                f"RANSAC: Removing anomalies from training dataset. "
                f"Score threshold {self.ransac_thresh[cur_ransac_pass]}."
            )

            masking_strategy = SimpleMaskingStrategy(
                score_threshold=self.ransac_thresh[cur_ransac_pass],
                anomaly_history_size=10,
                num_points_to_accept=14,
            )
            masking_state: Dict[str, Any] = {}
            with self.trainer.ctx:
                predictor = train_output.predictor

                # go through each time series in the dataset
                for i, d in enumerate(training_data):
                    data_series = to_pandas(d, freq=d["freq"])

                    # initialize data and delayed data buffer
                    delayed_data = data_series[: self.lead_time]
                    data_series = data_series[self.lead_time :]

                    # initialize predictor state
                    state = self.get_state_initializer()()

                    chunk_cnt = 0
                    while len(data_series) > 0:
                        assert len(delayed_data) == self.lead_time

                        # cut a chunk of lead_time length
                        chunk_cnt += 1
                        chunk = data_series[: self.lead_time]

                        # remove the chunk from data
                        data_series = data_series[len(chunk) :]

                        # forecasts
                        data_entry = {
                            "start": delayed_data.start,
                            "target": delayed_data[: len(chunk)].target,
                            **state,
                        }
                        forecasts = predictor.predict([data_entry])
                        forecast = next(iter(forecasts))
                        assert forecast.info is not None
                        state = forecast.info["predictor_state"]

                        # masking
                        (
                            masking_state,
                            maybe_masked,
                            is_masked,
                        ) = masking_strategy.mask(
                            masking_state, forecast, chunk
                        )

                        # update the delayed data buffer with the masked target
                        delayed_data = delayed_data.append(maybe_masked)
                        delayed_data = delayed_data[len(chunk) :]

                        # update the dataset with the masked target if there was a change
                        if sum(is_masked.target) > 0:
                            train_model = True
                            training_data.list_data[i]["target"][
                                chunk_cnt
                                * self.lead_time : chunk_cnt
                                * self.lead_time
                                + len(chunk)
                            ] = maybe_masked.target

                # if no anomalies found during ransac the dataset did not change
                # check if there are lower thresholds than the current that can catch anomalies
                # else end ransac
                if not train_model:
                    print(
                        f"RANSAC: No anomalies found in the dataset with "
                        f"score threshold {self.ransac_thresh[cur_ransac_pass]}."
                    )

                    prev_thresh = self.ransac_thresh[cur_ransac_pass]
                    cur_ransac_pass += 1
                    while cur_ransac_pass < num_ransac_passes:
                        if prev_thresh <= self.ransac_thresh[cur_ransac_pass]:
                            print(
                                f"RANSAC: Skip score threshold {self.ransac_thresh[cur_ransac_pass]}."
                            )
                            prev_thresh = self.ransac_thresh[cur_ransac_pass]
                            cur_ransac_pass += 1
                        else:
                            break
                else:
                    cur_ransac_pass += 1

                    # reset the model averaging parameters for the next training pass
                    if isinstance(
                        self.trainer.avg_strategy, IterationAveragingStrategy
                    ):
                        self.trainer.avg_strategy.averaging_started = False
                        self.trainer.avg_strategy.average_counter = 0
                        self.trainer.avg_strategy.averaged_model = None
                        self.trainer.avg_strategy.cached_model = None

        if train_model:
            train_output = self.train_model(
                training_data, validation_data, num_workers=num_workers
            )
        return train_output

    def train_model(
        self,
        training_data: Dataset,
        validation_data: Optional[Dataset] = None,
        num_workers: Optional[int] = None,
        num_prefetch: Optional[int] = None,
        shuffle_buffer_length: Optional[int] = None,
        **kwargs,
    ) -> TrainOutput:

        transformation = self.create_transformation()

        # ensure that the training network is created within the same MXNet
        # context as the one that will be used during training
        with self.trainer.ctx:
            trained_net = self.create_training_network()

        input_names = get_hybrid_forward_input_names(trained_net)
        data_preprocessing_transform = Chain(
            transformation.transformations[:-1]
        )
        assert isinstance(
            transformation.transformations[-1], AnomalyScoringSplitter
        )
        split_and_select = Chain(
            [transformation.transformations[-1], SelectFields(input_names)]
        )

        transformed_dataset = TransformedDataset(
            training_data, data_preprocessing_transform
        )

        training_data_loader = TrainDataLoader(
            dataset=Cached(transformed_dataset)
            if self.cache_data
            else transformed_dataset,
            transform=split_and_select,
            batch_size=self.batch_size,
            stack_fn=partial(batchify, ctx=self.trainer.ctx, dtype=self.dtype),
            num_workers=num_workers,
            num_prefetch=num_prefetch,
            shuffle_buffer_length=shuffle_buffer_length,
            **kwargs,
        )

        validation_data_loader = None

        if validation_data is not None:
            transformed_dataset = TransformedDataset(
                validation_data, data_preprocessing_transform
            )

            validation_data_loader = ValidationDataLoader(
                dataset=Cached(transformed_dataset)
                if self.cache_data
                else transformed_dataset,
                transform=split_and_select,
                batch_size=self.batch_size,
                stack_fn=partial(
                    batchify, ctx=self.trainer.ctx, dtype=self.dtype
                ),
                num_workers=num_workers,
                num_prefetch=num_prefetch,
                **kwargs,
            )

        self.trainer(
            net=trained_net,
            train_iter=training_data_loader,
            validation_iter=validation_data_loader,
        )

        with self.trainer.ctx:
            # ensure that the prediction network is created within the same MXNet
            # context as the one that was used during training
            return TrainOutput(
                transformation=transformation,
                trained_net=trained_net,
                predictor=self.create_predictor(transformation, trained_net),
            )

    def get_state_initializer(
        self, fsc: Optional[List[float]] = None
    ) -> StateInitializer:
        state_dict = {
            NETWORK_STATE_KEY: [
                np.zeros(
                    shape=(self.num_layers, self.hidden_size), dtype=np.float32
                )
            ],
            SCALE_STATE_KEY: None,
            LAGS_STATE_KEY: None,
        }

        for lag in self.lags[1:]:
            state_dict.update({lag.lag_state_field: None})
        if self.use_feat_static_cat:
            state_dict.update({FSC_STATE_KEY: fsc})

        return StateInitializer.from_state(state_dict)
