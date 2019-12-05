# Standard library imports
from typing import List, Optional

# Third-party imports
from gluonts.distribution import DistributionOutput
from gluonts.distribution.lowrank_gp import LowrankGPOutput
from mxnet.gluon import HybridBlock

# First-party imports
from gluonts.core.component import validated
from gluonts.model.deepvar._estimator import (
    get_lags_for_frequency,
    time_features_from_frequency_str,
)
from gluonts.model.estimator import GluonEstimator
from gluonts.model.predictor import Predictor, RepresentableBlockPredictor
from gluonts.support.util import copy_parameters
from gluonts.time_feature import TimeFeature
from gluonts.trainer import Trainer
from gluonts.transform import (
    AddObservedValuesIndicator,
    AddTimeFeatures,
    AsNumpyArray,
    Chain,
    ExpectedNumInstanceSampler,
    InstanceSplitter,
    SetFieldIfNotPresent,
    Transformation,
    VstackFeatures,
    ExpandDimArray,
    OneHotIndicator,
    SampleTargetDim,
    GaussianCopula,
    RenameFields,
    gaussian_copula_forward_transform,
)

# Relative imports
from gluonts.transform import FieldName
from ._network import GPVARPredictionNetwork, GPVARTrainingNetwork


class GPVAREstimator(GluonEstimator):
    @validated()
    def __init__(
        self,
        freq: str,
        prediction_length: int,
        target_dim: int,
        # number of dimension to sample at training time
        target_dim_sample: Optional[int] = None,
        distr_output: Optional[DistributionOutput] = None,
        trainer: Trainer = Trainer(),
        context_length: Optional[int] = None,
        num_layers: int = 2,
        num_cells: int = 40,
        cell_type: str = "lstm",
        num_eval_samples: int = 100,
        dropout_rate: float = 0.1,
        scaling: bool = True,
        pick_incomplete: bool = False,
        lags_seq: Optional[List[int]] = None,
        shuffle_target_dim: bool = True,
        time_features: Optional[List[TimeFeature]] = None,
        conditioning_length: int = 100,
        use_copula: bool = False,
    ) -> None:
        super().__init__(trainer=trainer)

        assert (
            prediction_length > 0
        ), "The value of `prediction_length` should be > 0"
        assert (
            context_length is None or context_length > 0
        ), "The value of `context_length` should be > 0"
        assert num_layers > 0, "The value of `num_layers` should be > 0"
        assert num_cells > 0, "The value of `num_cells` should be > 0"
        assert (
            num_eval_samples > 0
        ), "The value of `num_eval_samples` should be > 0"
        assert dropout_rate >= 0, "The value of `dropout_rate` should be >= 0"

        if distr_output is not None:
            self.distr_output = distr_output
        else:
            self.distr_output = LowrankGPOutput(rank=2)
        self.freq = freq
        self.context_length = (
            context_length if context_length is not None else prediction_length
        )
        self.prediction_length = prediction_length
        self.target_dim = target_dim
        self.target_dim_sample = (
            target_dim
            if target_dim_sample is None
            else min(target_dim_sample, target_dim)
        )
        self.shuffle_target_dim = shuffle_target_dim
        self.num_layers = num_layers
        self.num_cells = num_cells
        self.cell_type = cell_type
        self.num_sample_paths = num_eval_samples
        self.dropout_rate = dropout_rate

        self.lags_seq = (
            lags_seq
            if lags_seq is not None
            else get_lags_for_frequency(freq_str=freq)
        )
        self.time_features = (
            time_features
            if time_features is not None
            else time_features_from_frequency_str(self.freq)
        )

        self.history_length = self.context_length + max(self.lags_seq)
        self.pick_incomplete = pick_incomplete
        self.scaling = scaling
        self.conditioning_length = conditioning_length
        self.use_copula = use_copula
        if self.use_copula:
            self.output_transform = gaussian_copula_forward_transform
        else:
            self.output_transform = None

    def create_transformation(self) -> Transformation:
        def copula_transformation(use_copula: bool) -> Transformation:
            if use_copula:
                return GaussianCopula(
                    target_field=FieldName.TARGET,
                    observed_values_field=FieldName.OBSERVED_VALUES,
                    max_context_length=self.conditioning_length,
                    target_dim=self.target_dim,
                )
            else:
                return RenameFields(
                    {
                        f"past_{FieldName.TARGET}": f"past_{FieldName.TARGET}_cdf",
                        f"future_{FieldName.TARGET}": f"future_{FieldName.TARGET}_cdf",
                    }
                )

        return Chain(
            [
                AsNumpyArray(
                    field=FieldName.TARGET,
                    expected_ndim=1 + len(self.distr_output.event_shape),
                ),
                # maps the target to (1, T) if the target data is uni dimensional
                ExpandDimArray(
                    field=FieldName.TARGET,
                    axis=0 if self.distr_output.event_shape[0] == 1 else None,
                ),
                AddObservedValuesIndicator(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.OBSERVED_VALUES,
                ),
                AddTimeFeatures(
                    start_field=FieldName.START,
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_TIME,
                    time_features=self.time_features,
                    pred_length=self.prediction_length,
                ),
                VstackFeatures(
                    output_field=FieldName.FEAT_TIME,
                    input_fields=[FieldName.FEAT_TIME],
                ),
                SetFieldIfNotPresent(
                    field=FieldName.FEAT_STATIC_CAT, value=[0.0]
                ),
                OneHotIndicator(
                    field_name="target_dimensions",
                    target_field=FieldName.TARGET,
                ),
                AsNumpyArray(field=FieldName.FEAT_STATIC_CAT, expected_ndim=1),
                InstanceSplitter(
                    target_field=FieldName.TARGET,
                    is_pad_field=FieldName.IS_PAD,
                    start_field=FieldName.START,
                    forecast_start_field=FieldName.FORECAST_START,
                    train_sampler=ExpectedNumInstanceSampler(num_instances=1),
                    past_length=self.history_length,
                    future_length=self.prediction_length,
                    time_series_fields=[
                        FieldName.FEAT_TIME,
                        FieldName.OBSERVED_VALUES,
                    ],
                    pick_incomplete=self.pick_incomplete,
                ),
                copula_transformation(self.use_copula),
                SampleTargetDim(
                    field_name="target_dimensions",
                    target_field=FieldName.TARGET + "_cdf",
                    num_samples=self.target_dim_sample,
                    shuffle=self.shuffle_target_dim,
                ),
            ]
        )

    def create_training_network(self) -> GPVARTrainingNetwork:
        return GPVARTrainingNetwork(
            target_dim=self.target_dim,
            target_dim_sample=self.target_dim_sample,
            num_layers=self.num_layers,
            num_cells=self.num_cells,
            cell_type=self.cell_type,
            history_length=self.history_length,
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            dropout_rate=self.dropout_rate,
            lags_seq=self.lags_seq,
            scaling=self.scaling,
            distr_output=self.distr_output,
            conditioning_length=self.conditioning_length,
        )

    def create_predictor(
        self, transformation: Transformation, trained_network: HybridBlock
    ) -> Predictor:
        prediction_network = GPVARPredictionNetwork(
            target_dim=self.target_dim,
            target_dim_sample=self.target_dim,
            num_sample_paths=self.num_sample_paths,
            num_layers=self.num_layers,
            num_cells=self.num_cells,
            cell_type=self.cell_type,
            history_length=self.history_length,
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            dropout_rate=self.dropout_rate,
            lags_seq=self.lags_seq,
            scaling=self.scaling,
            distr_output=self.distr_output,
            conditioning_length=self.conditioning_length,
        )

        copy_parameters(trained_network, prediction_network)

        return RepresentableBlockPredictor(
            input_transform=transformation,
            prediction_net=prediction_network,
            batch_size=self.trainer.batch_size,
            freq=self.freq,
            prediction_length=self.prediction_length,
            ctx=self.trainer.ctx,
            output_transform=self.output_transform,
        )
