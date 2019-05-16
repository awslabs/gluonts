# Standard library imports
import logging
import re
from typing import Dict, List, Optional

# Third-party imports
import mxnet as mx
import numpy as np

# First-party imports
from gluonts import transform
from gluonts.core.component import validated
from gluonts.dataset.common import Dataset
from gluonts.dataset.loader import TrainDataLoader
from gluonts.model.estimator import GluonEstimator
from gluonts.model.predictor import Predictor, RepresentableBlockPredictor
from gluonts.model.wavenet._network import WaveNet, WaveNetSampler
from gluonts.support.util import (
    copy_parameters,
    get_hybrid_forward_input_names,
)
from gluonts.time_feature.lag import time_features_from_frequency_str
from gluonts.trainer import Trainer
from gluonts.transform import (
    AddAgeFeature,
    AddObservedValuesIndicator,
    AddTimeFeatures,
    AsNumpyArray,
    Chain,
    DataEntry,
    ExpectedNumInstanceSampler,
    FieldName,
    InstanceSplitter,
    SetFieldIfNotPresent,
    SimpleTransformation,
    VstackFeatures,
)


class QuantizeScaled(SimpleTransformation):
    '''
    Rescale and quantize the target variable.

    Requires
      past_target and future_target fields.

    The mean absolute value of the past_target is used to rescale past_target and future_target.
    Then the bin_edges are used to quantize the rescaled target.

    The calculated scale is included as a new field "scale"
    '''

    def __init__(
        self,
        bin_edges: np.ndarray,
        past_target: str,
        future_target: str,
        scale: str = 'scale',
    ):
        self.bin_edges = bin_edges
        self.future_target = future_target
        self.past_target = past_target
        self.scale = scale

    def transform(self, data: DataEntry) -> DataEntry:
        p = data[self.past_target]
        m = np.mean(np.abs(p))
        scale = m if m > 0 else 1.0
        data[self.future_target] = np.digitize(
            data[self.future_target] / scale, bins=self.bin_edges, right=False
        )
        data[self.past_target] = np.digitize(
            data[self.past_target] / scale, bins=self.bin_edges, right=False
        )
        data[self.scale] = np.array([scale])
        return data


def _get_seasonality(freq: str, seasonality_dict: Optional[Dict]) -> int:
    match = re.match(r'(\d*)(\w+)', freq)
    assert match, "Cannot match freq regex"
    multiple, base_freq = match.groups()
    multiple = int(multiple) if multiple else 1
    sdict = (
        seasonality_dict
        if seasonality_dict
        else {'H': 7 * 24, 'D': 7, 'W': 52, 'M': 12, 'B': 7 * 5}
    )
    seasonality = sdict[base_freq]
    if seasonality % multiple != 0:
        logging.warning(
            f'multiple {multiple} does not divide base seasonality {seasonality}.'
            f'Falling back to seasonality 1'
        )
        return 1
    return seasonality // multiple


class WaveNetEstimator(GluonEstimator):
    @validated()
    def __init__(
        self,
        freq: str,
        prediction_length: int,
        trainer: Trainer = Trainer(
            learning_rate=0.01,
            epochs=200,
            num_batches_per_epoch=50,
            hybridize=False,
        ),
        num_eval_samples: int = 200,
        cardinality: List[int] = [1],
        embedding_dimension: int = 5,
        num_bins: int = 1024,
        hybridize_prediction_net: bool = False,
        n_residue=24,
        n_skip=32,
        dilation_depth: Optional[int] = None,
        n_stacks: int = 1,
        temperature: float = 1.0,
        act_type: str = 'elu',
    ) -> None:
        """
        Model with Wavenet architecture and quantized target.

        :param freq:
        :param prediction_length:
        :param trainer:
        :param num_eval_samples:
        :param cardinality:
        :param embedding_dimension:
        :param num_bins: Number of bins used for quantization of signal
        :param hybridize_prediction_net:
        :param n_residue: Number of residual channels in wavenet architecture
        :param n_skip: Number of skip channels in wavenet architecture
        :param dilation_depth: number of dilation layers in wavenet architecture.
          If set to None, dialation_depth is set such that the receptive length is at least as long as typical
          seasonality for the frequency and at least 2 * prediction_length.
        :param n_stacks: Number of dilation stacks in wavenet architecture
        :param temperature: Temparature used for sampling from softmax distribution.
          For temperature = 1.0 sampling is according to estimated probability.
        :param act_type: Activation type used after before output layer.
          Can be any of
              'elu', 'relu', 'sigmoid', 'tanh', 'softrelu', 'softsign'
        """

        super().__init__(trainer=trainer)

        self.freq = freq
        self.prediction_length = prediction_length
        self.num_eval_samples = num_eval_samples
        self.cardinality = cardinality
        self.embedding_dimension = embedding_dimension
        self.num_bins = num_bins
        self.hybridize_prediction_net = hybridize_prediction_net

        self.n_residue = n_residue
        self.n_skip = n_skip
        self.n_stacks = n_stacks
        self.temperature = temperature
        self.act_type = act_type

        seasonality = _get_seasonality(
            self.freq, {'H': 7 * 24, 'D': 7, 'W': 52, 'M': 12, 'B': 7 * 5}
        )
        goal_receptive_length = max(seasonality, 2 * self.prediction_length)
        if dilation_depth is None:
            d = 1
            while (
                WaveNet.get_receptive_field(
                    dilation_depth=d, n_stacks=n_stacks
                )
                < goal_receptive_length
            ):
                d += 1
            self.dilation_depth = d
        else:
            self.dilation_depth = dilation_depth
        self.context_length = WaveNet.get_receptive_field(
            dilation_depth=self.dilation_depth, n_stacks=n_stacks
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(
            f'Using dilation depth {self.dilation_depth} and receptive field length {self.context_length}'
        )

    def train(self, training_data: Dataset) -> Predictor:
        has_negative_data = any(np.any(d['target'] < 0) for d in training_data)
        low = -10.0 if has_negative_data else 0
        high = 10.0
        bin_centers = np.linspace(low, high, self.num_bins)
        bin_edges = np.concatenate(
            [[-1e20], (bin_centers[1:] + bin_centers[:-1]) / 2.0, [1e20]]
        )

        transformation = self.create_transformation(bin_edges)

        transformation.estimate(iter(training_data))

        training_data_loader = TrainDataLoader(
            dataset=training_data,
            transform=transformation,
            batch_size=self.trainer.batch_size,
            num_batches_per_epoch=self.trainer.num_batches_per_epoch,
            ctx=self.trainer.ctx,
        )

        # ensure that the training network is created within the same MXNet
        # context as the one that will be used during training
        with self.trainer.ctx:
            trained_net = WaveNet(**self._get_wavenet_args(bin_centers))

        self.trainer(
            net=trained_net,
            input_names=get_hybrid_forward_input_names(trained_net),
            train_iter=training_data_loader,
        )

        # ensure that the prediction network is created within the same MXNet
        # context as the one that was used during training
        with self.trainer.ctx:
            return self.create_predictor(
                transformation, trained_net, bin_centers
            )

    def create_transformation(
        self, bin_edges: np.ndarray
    ) -> transform.Transformation:
        return Chain(
            [
                AsNumpyArray(field=FieldName.TARGET, expected_ndim=1),
                AddObservedValuesIndicator(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.OBSERVED_VALUES,
                ),
                AddTimeFeatures(
                    start_field=FieldName.START,
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_TIME,
                    time_features=time_features_from_frequency_str(self.freq),
                    pred_length=self.prediction_length,
                ),
                AddAgeFeature(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_AGE,
                    pred_length=self.prediction_length,
                ),
                VstackFeatures(
                    output_field=FieldName.FEAT_TIME,
                    input_fields=[FieldName.FEAT_TIME, FieldName.FEAT_AGE],
                ),
                SetFieldIfNotPresent(
                    field=FieldName.FEAT_STATIC_CAT, value=[0.0]
                ),
                AsNumpyArray(field=FieldName.FEAT_STATIC_CAT, expected_ndim=1),
                InstanceSplitter(
                    target_field=FieldName.TARGET,
                    is_pad_field=FieldName.IS_PAD,
                    start_field=FieldName.START,
                    forecast_start_field=FieldName.FORECAST_START,
                    train_sampler=ExpectedNumInstanceSampler(num_instances=1),
                    past_length=self.context_length,
                    future_length=self.prediction_length,
                    output_NTC=False,
                    time_series_fields=[
                        FieldName.FEAT_TIME,
                        FieldName.OBSERVED_VALUES,
                    ],
                ),
                QuantizeScaled(
                    bin_edges=bin_edges,
                    future_target='future_target',
                    past_target='past_target',
                ),
            ]
        )

    def _get_wavenet_args(self, bin_centers):
        return dict(
            n_residue=self.n_residue,
            n_skip=self.n_skip,
            dilation_depth=self.dilation_depth,
            n_stacks=self.n_stacks,
            act_type=self.act_type,
            cardinality=self.cardinality,
            embedding_dimension=self.embedding_dimension,
            bin_values=bin_centers,
            pred_length=self.prediction_length,
        )

    def create_predictor(
        self,
        transformation: transform.Transformation,
        trained_network: mx.gluon.HybridBlock,
        bin_values: np.ndarray,
    ) -> Predictor:

        prediction_network = WaveNetSampler(
            num_samples=self.num_eval_samples,
            temperature=self.temperature,
            **self._get_wavenet_args(bin_values),
        )

        # copy_parameters(net_source=trained_network, net_dest=prediction_network, ignore_extra=True)
        copy_parameters(
            net_source=trained_network, net_dest=prediction_network
        )

        return RepresentableBlockPredictor(
            input_transform=transformation,
            prediction_net=prediction_network,
            batch_size=self.trainer.batch_size,
            freq=self.freq,
            prediction_length=self.prediction_length,
            ctx=self.trainer.ctx,
        )
