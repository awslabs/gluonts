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

import logging
from distutils.util import strtobool
from typing import List

import mxnet as mx
import numpy as np
from pydantic import Field

from gluonts.core import serde
from gluonts.dataset.stat import calculate_dataset_statistics
from gluonts.mx.block.decoder import ForkingMLPDecoder
from gluonts.mx.block.encoder import (
    HierarchicalCausalConv1DEncoder,
    RNNEncoder,
)
from gluonts.mx.block.quantile_output import (
    QuantileOutput,
    IncrementalQuantileOutput,
)
from gluonts.mx.distribution import DistributionOutput
from gluonts.mx.trainer import Trainer

from ._forking_estimator import ForkingSeq2SeqEstimator


@serde.dataclass
class MQCNNEstimator(ForkingSeq2SeqEstimator):
    """
    An :class:`MQDNNEstimator` with a Convolutional Neural Network (CNN) as an
    encoder and a multi-quantile MLP as a decoder. Implements the MQ-CNN
    Forecaster, proposed in [WTN+17]_.

    Parameters
    ----------
    freq
        Time granularity of the data.
    prediction_length
        Length of the prediction, also known as 'horizon'.
    context_length
        Number of time units that condition the predictions, also known as
        'lookback period'. (default: 4 * prediction_length)
    use_past_feat_dynamic_real
        Whether to use the ``past_feat_dynamic_real`` field from the data.
        (default: False) Automatically inferred when creating the
        MQCNNEstimator with the `from_inputs` class method.
    use_feat_dynamic_real
        Whether to use the ``feat_dynamic_real`` field from the data.
        (default: False) Automatically inferred when creating the
        MQCNNEstimator with the `from_inputs` class method.
    use_feat_static_cat:
        Whether to use the ``feat_static_cat`` field from the data.
        (default: False) Automatically inferred when creating the
        MQCNNEstimator with the `from_inputs` class method.
    cardinality:
        Number of values of each categorical feature.
        This must be set if ``use_feat_static_cat == True`` (default: None)
        Automatically inferred when creating the MQCNNEstimator with the
        `from_inputs` class method.
    embedding_dimension:
        Dimension of the embeddings for categorical features.
        (default: [min(50, (cat+1)//2) for cat in cardinality])
    add_time_feature
        Adds a set of time features. (default: True)
    add_age_feature
        Adds an age feature. (default: False)
        The age feature starts with a small value at the start of the time
        series and grows over time.
    enable_encoder_dynamic_feature
        Whether the encoder should also be provided with the dynamic features
        (``age``, ``time`` and ``feat_dynamic_real`` if enabled respectively).
        (default: True)
    enable_decoder_dynamic_feature
        Whether the decoder should also be provided with the dynamic features
        (``age``, ``time`` and ``feat_dynamic_real`` if enabled respectively).
        (default: True)
        It makes sense to disable this, if you don't have ``feat_dynamic_real``
        for the prediction range.
    seed
        Will set the specified int seed for numpy anc MXNet if specified.
        (default: None)
    decoder_mlp_dim_seq
        The dimensionalities of the Multi Layer Perceptron layers of the
        decoder. (default: [30])
    channels_seq
        The number of channels (i.e. filters or convolutions) for each layer of
        the HierarchicalCausalConv1DEncoder. More channels usually correspond
        to better performance and larger network size.(default: [30, 30, 30])
    dilation_seq
        The dilation of the convolutions in each layer of the
        HierarchicalCausalConv1DEncoder. Greater numbers correspond to a
        greater receptive field of the network, which is usually better with
        longer context_length. (Same length as channels_seq)
        (default: [1, 3, 5])
    kernel_size_seq
        The kernel sizes (i.e. window size) of the convolutions in each layer
        of the HierarchicalCausalConv1DEncoder.
        (Same length as channels_seq) (default: [7, 3, 3])
    use_residual
        Whether the hierarchical encoder should additionally pass the unaltered
        past target to the decoder. (default: True)
    quantiles
        The list of quantiles that will be optimized for, and predicted by, the
        model. Optimizing for more quantiles than are of direct interest to you
        can result in improved performance due to a regularizing effect.
        (default: [0.025, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.975])
    distr_output
        DistributionOutput to use. Only one between `quantile` and
        `distr_output` can be set. (Default: None)
    trainer
        The GluonTS trainer to use for training. (default: Trainer())
    scaling
        Whether to automatically scale the target values. (default: False if
        quantile_output is used, True otherwise)
    scaling_decoder_dynamic_feature
        Whether to automatically scale the dynamic features for the decoder.
        (default: False)
    num_forking
        Decides how much forking to do in the decoder. 1 reduces to seq2seq and
        enc_len reduces to MQ-CNN.
    max_ts_len
        Returns the length of the longest time series in the dataset to be used
        in bounding context_length.
    is_iqf
        Determines whether to use IQF or QF. (default: True).
    batch_size
        The size of the batches to be used training and prediction.
    """

    freq: str = Field(...)
    prediction_length: int = Field(...)
    context_length: int = Field(None)
    use_past_feat_dynamic_real: bool = False
    use_feat_dynamic_real: bool = False
    use_feat_static_cat: bool = False
    cardinality: List[int] = Field(None)
    embedding_dimension: List[int] = Field(None)
    add_time_feature: bool = True
    add_age_feature: bool = False
    enable_encoder_dynamic_feature: bool = True
    enable_decoder_dynamic_feature: bool = True
    seed: int = Field(None)
    decoder_mlp_dim_seq: List[int] = Field(None)
    channels_seq: List[int] = Field(None)
    dilation_seq: List[int] = Field(None)
    kernel_size_seq: List[int] = Field(None)
    use_residual: bool = True
    quantiles: List[float] = Field(None)
    distr_output: DistributionOutput = Field(None)
    trainer: Trainer = Trainer()
    scaling: bool = Field(None)
    scaling_decoder_dynamic_feature: bool = False
    num_forking: int = Field(None)
    max_ts_len: int = Field(None)
    is_iqf: bool = True
    batch_size: int = 32

    def __post_init_post_parse__(self):
        assert (self.distr_output is None) or (self.quantiles is None)
        assert (
            self.prediction_length > 0
        ), f"Invalid prediction length: {self.prediction_length}."
        assert self.decoder_mlp_dim_seq is None or all(
            d > 0 for d in self.decoder_mlp_dim_seq
        ), "Elements of `mlp_hidden_dimension_seq` should be > 0"
        assert self.channels_seq is None or all(
            d > 0 for d in self.channels_seq
        ), "Elements of `channels_seq` should be > 0"
        assert self.dilation_seq is None or all(
            d > 0 for d in self.dilation_seq
        ), "Elements of `dilation_seq` should be > 0"
        # TODO: add support for kernel size=1
        assert self.kernel_size_seq is None or all(
            d > 1 for d in self.kernel_size_seq
        ), "Elements of `kernel_size_seq` should be > 0"
        assert self.quantiles is None or all(
            0 <= d <= 1 for d in self.quantiles
        ), "Elements of `quantiles` should be >= 0 and <= 1"

        self.decoder_mlp_dim_seq = (
            self.decoder_mlp_dim_seq
            if self.decoder_mlp_dim_seq is not None
            else [30]
        )
        self.channels_seq = (
            self.channels_seq
            if self.channels_seq is not None
            else [30, 30, 30]
        )
        self.dilation_seq = (
            self.dilation_seq if self.dilation_seq is not None else [1, 3, 9]
        )
        self.kernel_size_seq = (
            self.kernel_size_seq
            if self.kernel_size_seq is not None
            else [7, 3, 3]
        )
        self.quantiles = (
            self.quantiles
            if (self.quantiles is not None) or (self.distr_output is not None)
            else [0.025, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.975]
        )

        assert (
            len(self.channels_seq)
            == len(self.dilation_seq)
            == len(self.kernel_size_seq)
        ), (
            f"mismatch CNN configurations: {len(self.channels_seq)} vs. "
            f"{len(self.dilation_seq)} vs. {len(self.kernel_size_seq)}"
        )

        if self.seed:
            np.random.seed(self.seed)
            mx.random.seed(self.seed, self.trainer.ctx)

        # `use_static_feat` and `use_dynamic_feat` always True because network
        # always receives input; either from the input data or constants
        self.encoder = HierarchicalCausalConv1DEncoder(
            dilation_seq=self.dilation_seq,
            kernel_size_seq=self.kernel_size_seq,
            channels_seq=self.channels_seq,
            use_residual=self.use_residual,
            use_static_feat=True,
            use_dynamic_feat=True,
            prefix="encoder_",
        )

        self.decoder = ForkingMLPDecoder(
            dec_len=self.prediction_length,
            final_dim=self.decoder_mlp_dim_seq[-1],
            hidden_dimension_sequence=self.decoder_mlp_dim_seq[:-1],
            prefix="decoder_",
        )

        if not self.quantiles:
            self.quantile_output = None
        elif self.is_iqf:
            self.quantile_output = IncrementalQuantileOutput(self.quantiles)
        else:
            self.quantile_output = QuantileOutput(self.quantiles)

        # TODO: Honqing, check where to put
        super().__post_init_post_parse__()

    @classmethod
    def derive_auto_fields(cls, train_iter):
        stats = calculate_dataset_statistics(train_iter)

        return {
            "use_past_feat_dynamic_real": stats.num_past_feat_dynamic_real > 0,
            "use_feat_dynamic_real": stats.num_feat_dynamic_real > 0,
            "use_feat_static_cat": bool(stats.feat_static_cat),
            "cardinality": [len(cats) for cats in stats.feat_static_cat],
            "max_ts_len": stats.max_target_length,
        }

    @classmethod
    def from_inputs(cls, train_iter, **params):
        logger = logging.getLogger(__name__)
        logger.info(
            f"gluonts[from_inputs]: User supplied params set to {params}"
        )
        # auto_params usually include `use_feat_dynamic_real`,
        # `use_past_feat_dynamic_real`, `use_feat_static_cat` and
        # `cardinality`
        auto_params = cls.derive_auto_fields(train_iter)

        fields = [
            "use_feat_dynamic_real",
            "use_past_feat_dynamic_real",
            "use_feat_static_cat",
        ]
        # user defined arguments become implications
        for field in fields:
            if field in params.keys():
                is_params_field = (
                    params[field]
                    if type(params[field]) == bool
                    else strtobool(params[field])
                )
                if is_params_field and not auto_params[field]:
                    logger.warning(
                        f"gluonts[from_inputs]: {field} set to False since it"
                        " is not present in the data."
                    )
                    params[field] = False
                    if field == "use_feat_static_cat":
                        params["cardinality"] = None
                elif (
                    field == "use_feat_static_cat"
                    and not is_params_field
                    and auto_params[field]
                ):
                    params["cardinality"] = None

        # user specified 'params' will take precedence:
        params = {**auto_params, **params}
        logger.info(
            "gluonts[from_inputs]: use_past_feat_dynamic_real set to"
            f" '{params['use_past_feat_dynamic_real']}', use_feat_dynamic_real"
            f" set to '{params['use_feat_dynamic_real']}', and"
            f" use_feat_static_cat set to '{params['use_feat_static_cat']}'"
            f" with cardinality of '{params['cardinality']}'"
        )
        return cls.from_hyperparameters(**params)


@serde.dataclass
class MQRNNEstimator(ForkingSeq2SeqEstimator):
    """
    An :class:`MQDNNEstimator` with a Recurrent Neural Network (RNN) as an
    encoder and a multi-quantile MLP as a decoder.

    Implements the MQ-RNN Forecaster, proposed in [WTN+17]_.
    """

    prediction_length: int = Field(...)
    freq: str = Field(...)
    context_length: int = Field(None)
    decoder_mlp_dim_seq: List[int] = Field(None)
    trainer: Trainer = Trainer()
    quantiles: List[float] = Field(None)
    distr_output: DistributionOutput = Field(None)
    scaling: bool = Field(None)
    scaling_decoder_dynamic_feature: bool = False
    num_forking: int = Field(None)
    is_iqf: bool = True
    batch_size: int = 32

    def __post_init_post_parse__(self):
        assert (
            self.prediction_length > 0
        ), f"Invalid prediction length: {self.prediction_length}."
        assert self.decoder_mlp_dim_seq is None or all(
            d > 0 for d in self.decoder_mlp_dim_seq
        ), "Elements of `mlp_hidden_dimension_seq` should be > 0"
        assert self.quantiles is None or all(
            0 <= d <= 1 for d in self.quantiles
        ), "Elements of `quantiles` should be >= 0 and <= 1"

        self.decoder_mlp_dim_seq = (
            self.decoder_mlp_dim_seq
            if self.decoder_mlp_dim_seq is not None
            else [30]
        )
        self.quantiles = (
            self.quantiles
            if (self.quantiles is not None) or (self.distr_output is not None)
            else [0.025, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.975]
        )

        # `use_static_feat` and `use_dynamic_feat` always True because network
        # always receives input; either from the input data or constants
        self.encoder = RNNEncoder(
            mode="gru",
            hidden_size=50,
            num_layers=1,
            bidirectional=True,
            prefix="encoder_",
            use_static_feat=True,
            use_dynamic_feat=True,
        )

        self.decoder = ForkingMLPDecoder(
            dec_len=self.prediction_length,
            final_dim=self.decoder_mlp_dim_seq[-1],
            hidden_dimension_sequence=self.decoder_mlp_dim_seq[:-1],
            prefix="decoder_",
        )

        if not self.quantiles:
            self.quantile_output = None
        elif self.is_iqf:
            self.quantile_output = IncrementalQuantileOutput(self.quantiles)
        else:
            self.quantile_output = QuantileOutput(self.quantiles)

        super().__post_init_post_parse__()
