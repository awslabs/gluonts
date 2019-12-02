# Standard library imports
from typing import List, Optional, Callable, Tuple

# Third-party imports
import re
import pandas as pd
import numpy as np
from mxnet.gluon import HybridBlock

# First-party imports
from gluonts.core.component import validated
from gluonts.distribution import DistributionOutput, StudentTOutput
from gluonts.model.estimator import GluonEstimator
from gluonts.model.predictor import Predictor, RepresentableBlockPredictor
from gluonts.support.util import copy_parameters

from gluonts.time_feature import TimeFeature
from gluonts.dataset.field_names import FieldName

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
    TargetDimIndicator,
    CDFtoGaussianTransform,
    cdf_to_gaussian_forward_transform,
    RenameFields,
)

# Relative imports
from ._network import DeepVARPredictionNetwork, DeepVARTrainingNetwork


class FourrierDateFeatures(TimeFeature):
    @validated()
    def __init__(self, freq: str) -> None:
        # reocurring freq
        freqs = [
            "month",
            "day",
            "hour",
            "minute",
            "weekofyear",
            "weekday",
            "dayofweek",
            "dayofyear",
            "daysinmonth",
        ]

        assert freq in freqs
        self.freq = freq

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        values = getattr(index, self.freq)
        num_values = max(values) + 1
        steps = [x * 2.0 * np.pi / num_values for x in values]
        return np.vstack([np.cos(steps), np.sin(steps)])


def get_granularity(freq_str: str) -> Tuple[int, str]:
    """
    Splits a frequency string such as "7D" into the multiple 7 and the base
    granularity "D".
    Parameters
    ----------
    freq_str
        Frequency string of the form [multiple][granularity] such as "12H", "5min", "1D" etc.
    """
    freq_regex = r'\s*((\d+)?)\s*([^\d]\w*)'
    m = re.match(freq_regex, freq_str)
    assert m is not None, "Cannot parse frequency string: %s" % freq_str
    groups = m.groups()
    multiple = int(groups[1]) if groups[1] is not None else 1
    granularity = groups[2]
    return multiple, granularity


def time_features_from_frequency_str(freq_str: str) -> List[TimeFeature]:
    multiple, granularity = get_granularity(freq_str)

    features = {
        "M": ["weekofyear"],
        "W": ["daysinmonth", "weekofyear"],
        "D": ["dayofweek"],
        "B": ["dayofweek", "dayofyear"],
        "H": ["hour", "dayofweek"],
        "min": ["minute", "hour", "dayofweek"],
        "T": ["minute", "hour", "dayofweek"],
    }

    assert granularity in features, f"freq {granularity} not supported"

    feature_classes: List[TimeFeature] = [
        FourrierDateFeatures(freq=freq) for freq in features[granularity]
    ]
    return feature_classes


def get_lags_for_frequency(freq_str, num_lags=None):
    multiple, granularity = get_granularity(freq_str)

    if granularity == "M":
        lags = [[1, 12]]
    elif granularity == "D":
        lags = [[1, 7, 14]]
    elif granularity == "B":
        lags = [[1, 2]]
    elif granularity == "H":
        lags = [[1, 24, 168]]
    elif granularity == "min":
        lags = [[1, 4, 12, 24, 48]]
    else:
        lags = [[1]]

    # use less lags
    lags = [int(lag) for sub_list in lags for lag in sub_list]
    lags = sorted(list(set(lags)))
    return lags[:num_lags]


class DeepVAREstimator(GluonEstimator):
    @validated()
    def __init__(
        self,
        freq: str,
        prediction_length: int,
        target_dim: int,
        trainer: Trainer = Trainer(),
        context_length: Optional[int] = None,
        num_layers: int = 2,
        num_cells: int = 40,
        cell_type: str = "lstm",
        num_eval_samples: int = 100,
        dropout_rate: float = 0.1,
        cardinality: List[int] = [
            1
        ],  # TODO: we should infer this automatically somehow
        embedding_dimension: int = 5,
        distr_output: DistributionOutput = StudentTOutput(),
        scaling: bool = True,
        conditioning_length: int = 200,
        pick_incomplete: bool = False,
        lags_seq: Optional[List[int]] = None,
        time_features: Optional[List[TimeFeature]] = None,
        use_copula=False,
        **kwargs,
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
        assert all(
            [c > 0 for c in cardinality]
        ), "Elements of `cardinality` should be > 0"
        assert (
            embedding_dimension > 0
        ), "The value of `embedding_dimension` should be > 0"

        self.freq = freq
        self.context_length = (
            context_length if context_length is not None else prediction_length
        )
        self.prediction_length = prediction_length
        self.target_dim = target_dim
        self.distr_output = distr_output
        self.num_layers = num_layers
        self.num_cells = num_cells
        self.cell_type = cell_type
        self.num_sample_paths = num_eval_samples
        self.dropout_rate = dropout_rate
        self.cardinality = cardinality
        self.embedding_dimension = embedding_dimension
        self.conditioning_length = conditioning_length
        self.use_copula = use_copula

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

        if self.use_copula:
            self.output_transform: Optional[
                Callable
            ] = cdf_to_gaussian_forward_transform
        else:
            self.output_transform = None

    def create_transformation(self) -> Transformation:
        def copula_transformation(use_copula: bool) -> Transformation:
            if use_copula:
                return CDFtoGaussianTransform(
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
                TargetDimIndicator(
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
            ]
        )

    def create_training_network(self) -> DeepVARTrainingNetwork:
        return DeepVARTrainingNetwork(
            target_dim=self.target_dim,
            num_layers=self.num_layers,
            num_cells=self.num_cells,
            cell_type=self.cell_type,
            history_length=self.history_length,
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            distr_output=self.distr_output,
            dropout_rate=self.dropout_rate,
            cardinality=self.cardinality,
            embedding_dimension=self.embedding_dimension,
            lags_seq=self.lags_seq,
            scaling=self.scaling,
            conditioning_length=self.conditioning_length,
        )

    def create_predictor(
        self, transformation: Transformation, trained_network: HybridBlock
    ) -> Predictor:
        prediction_network = DeepVARPredictionNetwork(
            target_dim=self.target_dim,
            num_sample_paths=self.num_sample_paths,
            num_layers=self.num_layers,
            num_cells=self.num_cells,
            cell_type=self.cell_type,
            history_length=self.history_length,
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            distr_output=self.distr_output,
            dropout_rate=self.dropout_rate,
            cardinality=self.cardinality,
            embedding_dimension=self.embedding_dimension,
            lags_seq=self.lags_seq,
            scaling=self.scaling,
            conditioning_length=self.conditioning_length,
        )

        copy_parameters(trained_network, prediction_network)

        return RepresentableBlockPredictor(
            input_transform=transformation,
            prediction_net=prediction_network,
            # prediction_net=symbol_block,
            batch_size=self.trainer.batch_size,
            freq=self.freq,
            prediction_length=self.prediction_length,
            ctx=self.trainer.ctx,
            output_transform=self.output_transform,
        )
