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

from copy import deepcopy
from functools import partial
from typing import List, Tuple

from mxnet.gluon import HybridBlock
import numpy as np
import pandas as pd

from gluonts.core.component import Type, validated
from gluonts.dataset.common import DataEntry, Dataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.loader import (
    DataLoader,
    TrainDataLoader,
    ValidationDataLoader,
)
from gluonts.env import env
from gluonts.model.predictor import Predictor
from gluonts.mx.distribution import GaussianOutput
from gluonts.mx.model.estimator import GluonEstimator
from gluonts.mx.trainer import Trainer
from gluonts.mx.util import get_hybrid_forward_input_names
from gluonts.mx.util import copy_parameters
from gluonts.time_feature import (
    get_lags_for_frequency,
    time_features_from_frequency_str,
)
from gluonts.transform import (
    SelectFields,
    SimpleTransformation,
    Transformation,
)

from gluonts.nursery.temporal_hierarchical_forecasting.model.cop_deepar.gluonts_fixes import (
    batchify_with_dict,
    DeepAREstimatorForCOP,
    LOSS_FUNCTIONS,
    RepresentableBlockPredictorBatchifyWithDict,
)
from gluonts.nursery.temporal_hierarchical_forecasting.utils.common import (
    TEMPORAL_HIERARCHIES,
)
from gluonts.nursery.temporal_hierarchical_forecasting.model.cop_deepar._network import (
    COPDeepARTrainingNetwork,
    COPDeepARPredictionNetwork,
)


class AddTimeFeaturesAtAggregateLevels(SimpleTransformation):
    @validated()
    def __init__(
        self,
        agg_multiples: List[int],
        agg_estimators: List[GluonEstimator],
    ):
        self.agg_multiples = agg_multiples
        self.agg_estimators = agg_estimators

    def transform(self, data: DataEntry) -> DataEntry:
        past_length_bottom_ts = data[f"past_{FieldName.FEAT_TIME}"].shape[1]

        agg_features_dict = {}
        for i, (agg_multiple, agg_estimator) in enumerate(
            zip(self.agg_multiples, self.agg_estimators)
        ):
            (
                past_time_feat_agg,
                future_time_feat_agg,
            ) = self._get_time_features_agg_level(
                forecast_start=data[FieldName.FORECAST_START],
                past_length_bottom_ts=past_length_bottom_ts,
                agg_estimator=agg_estimator,
            )
            agg_features_dict[f"level_{i}"] = {
                "past_time_feat_agg": past_time_feat_agg,
                "future_time_feat_agg": future_time_feat_agg,
            }
            data.update({"agg_features_dict": agg_features_dict})
        return data

    @staticmethod
    def _get_time_features_agg_level(
        forecast_start: pd.Period,
        past_length_bottom_ts: int,
        agg_estimator: GluonEstimator,
    ) -> Tuple[np.ndarray, np.ndarray]:
        start = forecast_start - past_length_bottom_ts

        freq = agg_estimator.freq
        num_periods = (
            agg_estimator.history_length + agg_estimator.prediction_length
        )
        full_date_range = pd.period_range(
            start,
            periods=num_periods,
            freq=freq,
        )

        # shape: (T, num_features)
        full_time_feat = np.array(
            [
                feat_map(full_date_range)
                for feat_map in time_features_from_frequency_str(freq)
            ]
        ).T

        age_feature = np.log10(
            2.0 + np.arange(num_periods, dtype=agg_estimator.dtype)
        ).reshape((num_periods, 1))
        full_time_feat = np.hstack((full_time_feat, age_feature))

        past_time_feat = full_time_feat[: agg_estimator.history_length, :]
        future_time_feat = full_time_feat[agg_estimator.history_length :, :]

        return past_time_feat, future_time_feat


class COPDeepAREstimator(GluonEstimator):
    """
    Construct a COP estimator for temporal hierarchies.

    Parameters
    ----------
    freq
    prediction_length
    base_estimator_name
        Currently only `DeepAREstimator`
    base_estimator_hps
    do_reconciliation
        Flag to indicate if the samples at different aggregated levels should be reconciled. Useful for ablation study.
        Set this to False and check just learning multiple related tasks at the same time affects the accuracy at the
        bottom level; this is without doing any reconciliation.
    loss_function
        Training loss function to be used. Currently one of `crps_univariate` and `mse`.
    distr_output
    trainer
    lag_ub
    num_samples_for_loss
    dtype
    """

    @validated()
    def __init__(
        self,
        freq: str,
        prediction_length: int,
        base_estimator_name: str,
        base_estimator_hps: dict,
        use_gnn: bool = True,
        use_mlp: bool = True,
        adj_mat_option: str = "hierarchical",
        non_negative: bool = False,
        do_reconciliation: bool = True,
        loss_function: str = "crps_univariate",
        warmstart_epoch_frac: float = 0.5,
        trainer: Trainer = Trainer(hybridize=False),
        lag_ub: int = 1200,
        num_samples_for_loss: int = 200,
        return_forecasts_at_all_levels: bool = False,
        naive_reconciliation: bool = False,
        dtype: Type = np.float32,
    ) -> None:
        super().__init__(trainer=trainer, dtype=dtype)

        self.use_gnn = use_gnn
        self.use_mlp = use_mlp
        self.adj_mat_option = adj_mat_option
        self.do_reconciliation = do_reconciliation
        self.non_negative = non_negative
        self.warmstart_epoch_frac = warmstart_epoch_frac
        self.lag_ub = lag_ub
        self.num_samples_for_loss = num_samples_for_loss
        self.return_forecasts_at_all_levels = return_forecasts_at_all_levels
        self.naive_reconciliation = naive_reconciliation

        assert loss_function in LOSS_FUNCTIONS
        self.loss_function = loss_function

        self.context_length = base_estimator_hps.pop(
            "context_length", prediction_length
        )
        self.prediction_length = base_estimator_hps.pop(
            "prediction_length", prediction_length
        )
        self.freq = base_estimator_hps.pop("freq", freq)
        self.dtype = dtype

        self.temporal_hierarchy = TEMPORAL_HIERARCHIES[self.freq]
        self.base_estimator_type = eval(base_estimator_name)

        assert self.base_estimator_type == DeepAREstimatorForCOP

        if "distr_output" not in base_estimator_hps:
            base_estimator_hps["distr_output"] = GaussianOutput()

        print(f"Distribution output: {base_estimator_hps['distr_output']}")

        self.estimators = []
        for agg_multiple, freq_str in zip(
            self.temporal_hierarchy.agg_multiples,
            self.temporal_hierarchy.freq_strs,
        ):
            base_estimator_hps_agg = deepcopy(base_estimator_hps)
            # Following is a hack because gluonts does not allow setting lag_ub from outside!
            lags_seq = get_lags_for_frequency(
                freq_str=freq_str,
                lag_ub=lag_ub // agg_multiple,
            )

            # Remove lags that will not be available for reconciliation during inference.
            num_nodes = self.temporal_hierarchy.num_leaves // agg_multiple
            lags_seq = [lag for lag in lags_seq if lag >= num_nodes]

            # Hack to enforce correct serialization of lags_seq and history length
            # (only works when set in constructor).
            if agg_multiple != 1:
                estimator = self.base_estimator_type(
                    freq=freq_str,
                    context_length=self.context_length // agg_multiple,
                    lags_seq=lags_seq,
                    prediction_length=self.prediction_length // agg_multiple,
                    **base_estimator_hps_agg,
                )
            else:
                estimator = self.base_estimator_type(
                    freq=freq_str,
                    context_length=self.context_length // agg_multiple,
                    lags_seq=lags_seq,
                    prediction_length=self.prediction_length // agg_multiple,
                    **base_estimator_hps_agg,
                    # Load more data at the bottom level so that enough lags are available for the aggregated levels.
                    history_length=(
                        (self.context_length // agg_multiple) + max(lags_seq)
                    )
                    * 2,
                )

            self.estimators.append(estimator)

        self.agg_estimators = self.estimators[:-1]
        self.base_estimator = self.estimators[-1]

        self.agg_feature_adder = AddTimeFeaturesAtAggregateLevels(
            agg_multiples=self.temporal_hierarchy.agg_multiples[:-1],
            agg_estimators=self.agg_estimators,
        )

    def create_transformation(self) -> Transformation:
        # We use the base estimator to create the data transformation.
        return self.base_estimator.create_transformation()

    def create_training_data_loader(
        self,
        data: Dataset,
        **kwargs,
    ) -> DataLoader:
        input_names = get_hybrid_forward_input_names(COPDeepARTrainingNetwork)
        instance_splitter = self.base_estimator._create_instance_splitter(
            "training"
        )

        return TrainDataLoader(
            dataset=data,
            transform=instance_splitter
            + self.agg_feature_adder
            + SelectFields(input_names),
            batch_size=self.batch_size,
            stack_fn=partial(
                batchify_with_dict, ctx=self.trainer.ctx, dtype=self.dtype
            ),
            **kwargs,
        )

    def create_validation_data_loader(
        self,
        data: Dataset,
        **kwargs,
    ) -> DataLoader:
        input_names = get_hybrid_forward_input_names(COPDeepARTrainingNetwork)
        instance_splitter = self.base_estimator._create_instance_splitter(
            "validation"
        )
        return ValidationDataLoader(
            dataset=data,
            transform=instance_splitter
            + self.agg_feature_adder
            + SelectFields(input_names),
            batch_size=self.batch_size,
            stack_fn=partial(
                batchify_with_dict, ctx=self.trainer.ctx, dtype=self.dtype
            ),
        )

    def create_training_network(self) -> COPDeepARTrainingNetwork:
        return COPDeepARTrainingNetwork(
            estimators=self.estimators,
            prediction_length=self.prediction_length,
            temporal_hierarchy=self.temporal_hierarchy,
            num_batches_per_epoch=self.trainer.num_batches_per_epoch,
            epochs=self.trainer.epochs,
            warmstart_epoch_frac=self.warmstart_epoch_frac,
            num_samples_for_loss=self.num_samples_for_loss,
            use_gnn=self.use_gnn,
            use_mlp=self.use_mlp,
            adj_mat_option=self.adj_mat_option,
            do_reconciliation=self.do_reconciliation,
            non_negative=self.non_negative,
            naive_reconciliation=self.naive_reconciliation,
            loss_function=self.loss_function,
            dtype=self.dtype,
        )

    def create_predictor(
        self, transformation: Transformation, trained_network: HybridBlock
    ) -> Predictor:
        prediction_splitter = self.base_estimator._create_instance_splitter(
            "test"
        )

        prediction_network = COPDeepARPredictionNetwork(
            estimators=self.estimators,
            prediction_length=self.prediction_length,
            temporal_hierarchy=self.temporal_hierarchy,
            use_gnn=self.use_gnn,
            use_mlp=self.use_mlp,
            adj_mat_option=self.adj_mat_option,
            do_reconciliation=self.do_reconciliation,
            non_negative=self.non_negative,
            naive_reconciliation=self.naive_reconciliation,
            return_forecasts_at_all_levels=self.return_forecasts_at_all_levels,
            num_parallel_samples=self.base_estimator.num_parallel_samples,
            dtype=self.dtype,
        )

        copy_parameters(trained_network, prediction_network)
        return RepresentableBlockPredictorBatchifyWithDict(
            input_transform=transformation
            + prediction_splitter
            + self.agg_feature_adder,
            prediction_net=prediction_network,
            batch_size=self.batch_size,
            freq=self.freq,
            prediction_length=self.prediction_length,
            ctx=self.trainer.ctx,
            dtype=self.dtype,
        )
