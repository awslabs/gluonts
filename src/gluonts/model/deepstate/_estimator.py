# Standard library imports
from typing import List, Optional

# Third-party imports
from mxnet.gluon import HybridBlock

# First-party imports
from gluonts.core.component import validated
from gluonts.model.deepstate.issm import ISSM, CompositeISSM
from gluonts.time_feature.lag import (
    time_features_from_frequency_str,
    longest_period_from_frequency_str,
)
from gluonts.model.estimator import GluonEstimator
from gluonts.model.predictor import Predictor, RepresentableBlockPredictor
from gluonts.support.util import copy_parameters
from gluonts.time_feature.lag import (
    TimeFeature,
    time_features_from_frequency_str,
)
from gluonts.trainer import Trainer
from gluonts.transform import (
    FieldName,
    AddObservedValuesIndicator,
    AddTimeFeatures,
    AsNumpyArray,
    Chain,
    ExpandDimArray,
    ExpectedNumInstanceSampler,
    TestSplitSampler,
    CanonicalInstanceSplitter,
    Transformation,
    SetFieldIfNotPresent,
)

# Relative imports
from ._network import DeepStatePredictionNetwork, DeepStateTrainingNetwork

SEASON_INDICATORS_FIELD = "seasonal_indicators"


class DeepStateEstimator(GluonEstimator):
    """
    Construct a DeepState estimator.
    
    This implements the deep state space model described in
    [Rangapuram et. al. 2018]

    .. [Rangapuram et. al. 2018] Syama S. Rangapuram, Matthias W. Seeger,
    Jan Gasthaus, Lorenzo Stella, Yuyang Wang, Tim Januschowski.
    Deep State Space Models for Time Series Forecasting.
    In NeurIPS 31, pages 7785–7794, 2018.
    https://papers.nips.cc/paper/8004-deep-state-space-models-for-time-series-forecasting

    Parameters
    ----------
    freq
        Frequency of the data to train on and predict
    prediction_length
        Length of the prediction horizon
    add_trend
        Flag to indicate whether to include trend component in the
        state space model
    past_length
        Number of steps to unroll the RNN for before computing predictions.
        Set this to (at most) the length of the shortest time series in the dataset.
    trainer
        Trainer object to be used (default: Trainer())
    num_layers
        Number of RNN layers (default: 2)
    num_cells
        Number of RNN cells for each layer (default: 40)
    cell_type
        Type of recurrent cells to use (available: 'lstm' or 'gru';
        default: 'lstm')
    num_eval_samples
        Number of samples paths to draw when computing predictions
        (default: 100)
    dropout_rate
        Dropout regularization parameter (default: 0.1)
    cardinality
        Number of values of the each categorical feature (default: [1])
    embedding_dimension
        Dimension of the embeddings for categorical features (the same
        dimension is used for all embeddings, default: 5)
    scaling
        Whether to automatically scale the target values (default: true)
    lags_seq
        Indices of the lagged target values to use as inputs of the RNN
        (default: None, in which case these are automatically determined
        based on freq)
    time_features
        Time features to use as inputs of the RNN (default: None, in which
        case these are automatically determined based on freq)
    """

    @validated()
    def __init__(
        self,
        freq: str,
        prediction_length: int,
        add_trend: bool = False,
        past_length: Optional[int] = None,
        trainer: Trainer = Trainer(epochs=25),
        num_layers: int = 2,
        num_cells: int = 40,
        cell_type: str = "lstm",
        num_eval_samples: int = 100,
        dropout_rate: float = 0.1,
        use_feat_dynamic_real: bool = False,
        use_feat_static_cat: bool = False,
        cardinality: Optional[List[int]] = None,
        embedding_dimension: int = 20,
        issm: Optional[ISSM] = None,
        scaling: bool = True,
        time_features: Optional[List[TimeFeature]] = None,
    ) -> None:
        super().__init__(trainer=trainer)

        assert (
            prediction_length > 0
        ), "The value of `prediction_length` should be > 0"
        assert (
            past_length is None or past_length > 0
        ), "The value of `past_length` should be > 0"
        assert num_layers > 0, "The value of `num_layers` should be > 0"
        assert num_cells > 0, "The value of `num_cells` should be > 0"
        assert (
            num_eval_samples > 0
        ), "The value of `num_eval_samples` should be > 0"
        assert dropout_rate > 0, "The value of `dropout_rate` should be > 0"
        assert (
            cardinality is not None or not use_feat_static_cat
        ), "You must set `cardinality` if `use_feat_static_cat=True`"
        assert cardinality is None or [
            c > 0 for c in cardinality
        ], "Elements of `cardinality` should be > 0"
        assert (
            embedding_dimension > 0
        ), "The value of `embedding_dimension` should be > 0"

        self.freq = freq
        self.past_length = (
            past_length
            if past_length is not None
            else 4 * longest_period_from_frequency_str(freq)
        )
        self.prediction_length = prediction_length
        self.add_trend = add_trend
        self.num_layers = num_layers
        self.num_cells = num_cells
        self.cell_type = cell_type
        self.num_sample_paths = num_eval_samples
        self.scaling = scaling
        self.dropout_rate = dropout_rate
        self.use_feat_dynamic_real = use_feat_dynamic_real
        self.use_feat_static_cat = use_feat_static_cat
        self.cardinality = cardinality if use_feat_static_cat else [1]
        self.embedding_dimension = embedding_dimension

        self.issm = (
            issm
            if issm is not None
            else CompositeISSM.get_from_freq(freq, add_trend)
        )

        self.time_features = (
            time_features
            if time_features is not None
            else time_features_from_frequency_str(self.freq)
        )

    def create_transformation(self) -> Transformation:
        return Chain(
            [
                AsNumpyArray(field=FieldName.TARGET, expected_ndim=1),
                # gives target the (1, T) layout
                ExpandDimArray(field=FieldName.TARGET, axis=0),
                AddObservedValuesIndicator(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.OBSERVED_VALUES,
                ),
                AddTimeFeatures(
                    time_features=CompositeISSM.seasonal_features(self.freq),
                    pred_length=self.prediction_length,
                    start_field=FieldName.START,
                    target_field=FieldName.TARGET,
                    output_field=SEASON_INDICATORS_FIELD,
                ),
                AddTimeFeatures(
                    start_field=FieldName.START,
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_TIME,
                    time_features=self.time_features,
                    pred_length=self.prediction_length,
                ),
                SetFieldIfNotPresent(
                    field=FieldName.FEAT_STATIC_CAT, value=[0.0]
                ),
                AsNumpyArray(field=FieldName.FEAT_STATIC_CAT, expected_ndim=1),
                CanonicalInstanceSplitter(
                    target_field=FieldName.TARGET,
                    is_pad_field=FieldName.IS_PAD,
                    start_field=FieldName.START,
                    forecast_start_field=FieldName.FORECAST_START,
                    instance_sampler=TestSplitSampler(),
                    time_series_fields=[
                        FieldName.FEAT_TIME,
                        SEASON_INDICATORS_FIELD,
                        FieldName.OBSERVED_VALUES,
                    ],
                    allow_target_padding=True,
                    instance_length=self.past_length,
                    use_prediction_features=True,
                    prediction_length=self.prediction_length,
                ),
            ]
        )

    def create_training_network(self) -> DeepStateTrainingNetwork:
        return DeepStateTrainingNetwork(
            num_layers=self.num_layers,
            num_cells=self.num_cells,
            cell_type=self.cell_type,
            past_length=self.past_length,
            prediction_length=self.prediction_length,
            issm=self.issm,
            dropout_rate=self.dropout_rate,
            cardinality=self.cardinality,
            embedding_dimension=self.embedding_dimension,
            scaling=self.scaling,
        )

    def create_predictor(
        self, transformation: Transformation, trained_network: HybridBlock
    ) -> Predictor:
        prediction_network = DeepStatePredictionNetwork(
            num_sample_paths=self.num_sample_paths,
            num_layers=self.num_layers,
            num_cells=self.num_cells,
            cell_type=self.cell_type,
            past_length=self.past_length,
            prediction_length=self.prediction_length,
            issm=self.issm,
            dropout_rate=self.dropout_rate,
            cardinality=self.cardinality,
            embedding_dimension=self.embedding_dimension,
            scaling=self.scaling,
            params=trained_network.collect_params(),
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
