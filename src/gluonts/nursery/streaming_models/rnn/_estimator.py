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
import logging
import pickle
from collections import OrderedDict
from functools import partial
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd
from mxnet.gluon import HybridBlock

from gluonts.core.component import DType, validated
from gluonts.dataset.common import Dataset, ListDataset
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
from gluonts.mx.batchify import as_in_context, batchify
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
    AsNumpyArray,
    Chain,
    ExpectedNumInstanceSampler,
    MissingValueImputation,
    RemoveFields,
    RenameFields,
    SelectFields,
    SetField,
    SwapAxes,
    Transformation,
    TransformedDataset,
    VstackFeatures,
)

from ..forecast_generator import StatefulDistributionForecastGenerator
from ..masking import SimpleMaskingStrategy
from ..predictor import NETWORK_STATE_KEY
from ..transform import (
    AddShiftedTimestamp,
    AddStreamAggregateLags,
    AddStreamScale,
    CopyField,
    LeadtimeShifter,
    StreamingInstanceSplitter,
)
from ._defaults import (
    DEFAULT_LAGS_UB,
    DEFAULT_SMALL_LAGS,
    PRECISION,
    SUPPORTED_FREQS,
    RnnDefaults,
)
from ._network import StreamingRnnPredictNetwork, StreamingRnnTrainNetwork

logger = logging.getLogger(__name__)

LAGS_STATE_KEY = "s:lag"
SCALE_STATE_KEY = "s:scale"
FSC_STATE_KEY = "s:feat_static_cat"


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
        cache_data: bool = RnnDefaults.CACHE_DATA,
        cache_bytes_limit: int = RnnDefaults.CACHE_BYTES_LIMIT,
        normalization: str = "centeredscale",
        batch_size: int = RnnDefaults.BATCH_SIZE,
        dtype: DType = np.float32,
    ) -> None:
        super().__init__(trainer=trainer)

        assert lead_time > 0
        assert train_window_length > 0
        assert 0 <= skip_initial_window_pct < 1
        assert hidden_size > 0
        assert num_layers > 0
        assert alpha >= 0
        assert beta >= 0

        if cache_data:
            assert cache_bytes_limit > 0
        self.cache_data = cache_data
        self.cache_bytes_limit = cache_bytes_limit

        self.hybridize_prediction_net = hybridize_prediction_net
        self.freq = freq
        self.lead_time = lead_time
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
        self.normalization = normalization
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
                mean_field="mean",
                state_field=SCALE_STATE_KEY,
                minimum_value=1.0e-10,
                initial_scale=1.0e-10,
                alpha=0.002,
                normalization=self.normalization,
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
            # In LeadtimeShifter the time_series_fields are aligned with label_target
            LeadtimeShifter(
                lead_time=self.lead_time,
                target_field=FieldName.TARGET,
                time_series_fields=[
                    FieldName.FEAT_TIME,
                    FieldName.OBSERVED_VALUES,
                ],
            ),
            StreamingInstanceSplitter(
                target_field="label_target",
                is_pad_field=FieldName.IS_PAD,
                start_field=FieldName.START,
                forecast_start_field=FieldName.FORECAST_START,
                instance_sampler=ExpectedNumInstanceSampler(
                    num_instances=1.0,
                    min_past=self.train_window_length,
                    min_future=self.lead_time + 1,
                ),
                train_window_length=self.train_window_length,
                output_NTC=True,
                time_series_fields=[
                    "input_target",
                    FieldName.FEAT_TIME,
                    FieldName.OBSERVED_VALUES,
                    "lags",
                    "scale",
                    "mean",
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

    def create_transformation(self) -> Transformation:
        return Chain(
            self._data_transformation_steps() + self._training_batch_maker()
        )

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
        num_prefetch: Optional[int] = None,
        **kwargs,
    ) -> Predictor:
        assert isinstance(training_data, ListDataset)

        # subsample the datasets if needed when caching is used
        # needs to be done before ransac
        if self.cache_data:
            assert validation_data is None or isinstance(
                validation_data, ListDataset
            )
            training_data, validation_data = self._data_subsampling(
                training_data, validation_data
            )

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
                transform=predictor.input_transform,
                batch_size=predictor.batch_size,
                stack_fn=partial(
                    batchify, ctx=self.trainer.ctx, dtype=self.dtype
                ),
                num_workers=num_workers,
                num_prefetch=num_prefetch,
                **kwargs,
            )

            d = list(inf_loader)[0]
            return predictor.as_symbol_block_predictor(d)
        else:
            return predictor

    def _data_subsampling(
        self,
        training_data: ListDataset,
        validation_data: Optional[ListDataset],
    ) -> Tuple[ListDataset, Optional[ListDataset]]:
        def _calculate_memory_and_stats(training_data):
            total_bytes = 0
            cat_feat_dict: dict = OrderedDict()
            num_lags = sum(
                [len(l.valid_lags) for l in self.lags]
            )  # number of lags
            for i, d in enumerate(training_data):
                ts_len = len(d["target"])

                # expected size of each time series entry after transformation
                # overheads: 96 Bytes for 1D arrays and 112 for 2D arrays
                ts_size_bytes = (
                    (
                        (ts_len - self.lead_time) * PRECISION + 96
                    )  # input_target
                    + (
                        (ts_len - self.lead_time) * PRECISION + 96
                    )  # label_target
                    + (
                        (ts_len - self.lead_time) * PRECISION + 96
                    )  # observed_values
                    + ((ts_len - self.lead_time) * PRECISION + 96)  # time_feat
                    + (ts_len * PRECISION + 96)  # scale
                    + (num_lags * ts_len * PRECISION + 112)  # lags
                    + (
                        len(self.cardinality) * PRECISION + 96
                    )  # feat_static_cat
                    + 500  # approximately start field and dictionary object
                )

                total_bytes += ts_size_bytes

                # the cat_feat_dict is used for subsampling if needed
                # if the dataset has categorical features use them else create a common dummy one
                dict_key = (
                    tuple(d[FieldName.FEAT_STATIC_CAT])
                    if FieldName.FEAT_STATIC_CAT in d
                    else "0"
                )
                if dict_key not in cat_feat_dict:
                    cat_feat_dict[dict_key] = {
                        "ts_idx": [i],
                        "memory": [ts_size_bytes],
                    }
                else:
                    cat_feat_dict[dict_key]["ts_idx"].append(i)
                    cat_feat_dict[dict_key]["memory"].append(ts_size_bytes)

            return total_bytes, cat_feat_dict

        def _ts_selection(cat_feat_dict, memory_alloc_per_cat):
            selected_ts_indexes: List[int] = []
            total_selected_memory = 0
            for i, v in enumerate(cat_feat_dict.values()):
                # remove entries that are above the memory limit
                idx = [
                    val
                    for j, val in enumerate(v["ts_idx"])
                    if v["memory"][j] <= memory_alloc_per_cat[i]
                ]
                memory = [
                    val
                    for j, val in enumerate(v["memory"])
                    if v["memory"][j] <= memory_alloc_per_cat[i]
                ]

                # select time series from category
                # instead of cumsum a more optimal approach is knapsack but its complexity is prohibitive
                num_selected_ts = sum(
                    np.cumsum(memory) <= memory_alloc_per_cat[i]
                )
                selected_ts_indexes.extend(idx[:num_selected_ts])
                total_selected_memory += sum(memory[:num_selected_ts])

            return selected_ts_indexes, total_selected_memory

        def _squash_ts_categories(cat_feat_dict):
            ts_idx = []
            memory = []
            for c in cat_feat_dict.values():
                ts_idx.extend(c["ts_idx"])
                memory.extend(c["memory"])
            cat_feat_dict = {"0": {"ts_idx": ts_idx, "memory": memory}}

            return cat_feat_dict

        if validation_data is not None:
            assert len(training_data) == len(validation_data)

        total_bytes, cat_feat_dict = _calculate_memory_and_stats(training_data)
        if total_bytes < self.cache_bytes_limit:
            logger.info(
                "Transformed dataset fits into memory. No subsampling applied."
            )
            return training_data, validation_data

        # 1) the following subsampling method selects time series from each category proportional
        # to the memory (which is proportional to the timestamps) that each category consumes
        # 2) it ensures that the selected time series do not exceed the allocated memory
        # per category after the subsampling
        # 3) the selection based on categorical features is done regardless of
        # the use of the feat_static_cat field in the actual model
        alpha = self.cache_bytes_limit / total_bytes  # shrinkage pct
        logger.info(
            f"Transformed dataset does not fit into memory. Reducing it by {round((1- alpha) * 100, 2)}%."
        )

        # if there are too many categories, e.g., one per time series, we cannot subsample
        # from each category - in this case treat all time series as once category
        if (
            len(cat_feat_dict) > len(training_data) // 2
        ):  # at least two time series per category on average
            cat_feat_dict = _squash_ts_categories(cat_feat_dict)
            # if we do not see all categories during training inference will fail
            # do not use categorical features at all in this case
            self.use_feat_static_cat = False

        # memory limit per category
        memory_alloc_per_cat = alpha * np.array(
            [sum(cat_feat_dict[k]["memory"]) for k in cat_feat_dict.keys()]
        )

        selected_ts_indexes, total_selected_memory = _ts_selection(
            cat_feat_dict, memory_alloc_per_cat
        )

        # there is always a possibility of a big memory gap between the selected time series
        # and the memory limit that can be caused by the memory limits per category
        # in this case again treat all categories as one and resample
        # this partially addresses the issue of an empty subset of time series if it is caused by category limits
        if total_selected_memory / sum(memory_alloc_per_cat) < 0.5:
            if len(cat_feat_dict) > 1:
                cat_feat_dict = _squash_ts_categories(cat_feat_dict)
                self.use_feat_static_cat = False

                memory = alpha * np.array([sum(cat_feat_dict["0"]["memory"])])
                selected_ts_indexes, total_selected_memory = _ts_selection(
                    cat_feat_dict, memory
                )

        assert selected_ts_indexes, "No single time series fits into memory."
        logger.info(f"Subsampled {len(selected_ts_indexes)} time series.")

        # subsample the datasets using the selected time series indexes
        selected_training_entries = []
        for i, d in enumerate(training_data):
            if i in selected_ts_indexes:
                selected_training_entries.append(d)
        subsampled_training_data = ListDataset(
            selected_training_entries, freq=self.freq
        )

        selected_validation_entries = None
        if validation_data is not None:
            selected_validation_entries = []
            for i, d in enumerate(validation_data):
                if i in selected_ts_indexes:
                    selected_validation_entries.append(d)
            subsampled_validation_data = ListDataset(
                selected_validation_entries, freq=self.freq
            )

        return subsampled_training_data, subsampled_validation_data

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
                logger.info("RANSAC: Model training.")
                train_output = self.train_model(
                    training_data, validation_data, num_workers=num_workers
                )
                train_model = False
            else:
                logger.info("RANSAC: Skip model training.")

            logger.info(
                f"RANSAC: Removing anomalies from training dataset. "
                f"Score threshold {self.ransac_thresh[cur_ransac_pass]}."
            )

            masking_strategy = SimpleMaskingStrategy(
                score_threshold=self.ransac_thresh[cur_ransac_pass],
                anomaly_history_size=50,
                num_points_to_accept=100,
                mask_missing_values=False,
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

                    # check if static_feature exists in data
                    if FieldName.FEAT_STATIC_CAT in d:
                        state[FSC_STATE_KEY] = np.array(
                            d[FieldName.FEAT_STATIC_CAT]
                        )
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
                            masking_state,
                            forecast,
                            chunk,
                            state[SCALE_STATE_KEY][2],
                        )

                        # update the delayed data buffer with the masked target
                        delayed_data += maybe_masked
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
                    logger.info(
                        f"RANSAC: No anomalies found in the dataset with "
                        f"score threshold {self.ransac_thresh[cur_ransac_pass]}."
                    )

                    prev_thresh = self.ransac_thresh[cur_ransac_pass]
                    cur_ransac_pass += 1
                    while cur_ransac_pass < num_ransac_passes:
                        if prev_thresh <= self.ransac_thresh[cur_ransac_pass]:
                            logger.info(
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
        assert isinstance(transformation, Chain)

        input_names = get_hybrid_forward_input_names(StreamingRnnTrainNetwork)

        pre_split_tfs = Chain(transformation.transformations[:-1])
        splitter = transformation.transformations[-1]
        assert isinstance(splitter, StreamingInstanceSplitter)

        transformed_dataset = TransformedDataset(training_data, pre_split_tfs)

        training_data_loader = TrainDataLoader(
            dataset=Cached(transformed_dataset)
            if self.cache_data
            else transformed_dataset,
            transform=splitter + SelectFields(input_names),
            batch_size=self.batch_size,
            stack_fn=partial(
                batchify,
                ctx=self.trainer.ctx,
                dtype=self.dtype,
            ),
            num_workers=num_workers,
            num_prefetch=num_prefetch,
            shuffle_buffer_length=shuffle_buffer_length,
            decode_fn=partial(as_in_context, ctx=self.trainer.ctx),
            **kwargs,
        )

        validation_data_loader = None
        if validation_data is not None:
            transformed_dataset = TransformedDataset(
                validation_data, pre_split_tfs
            )

            validation_data_loader = ValidationDataLoader(
                dataset=Cached(transformed_dataset)
                if self.cache_data
                else transformed_dataset,
                transform=splitter + SelectFields(input_names),
                batch_size=self.batch_size,
                stack_fn=partial(
                    batchify,
                    ctx=self.trainer.ctx,
                    dtype=self.dtype,
                ),
                num_workers=num_workers,
                num_prefetch=num_prefetch,
                decode_fn=partial(as_in_context, ctx=self.trainer.ctx),
                **kwargs,
            )

        # ensure that the training network is created within the same MXNet
        # context as the one that will be used during training
        with self.trainer.ctx:
            trained_net = self.create_training_network()

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
