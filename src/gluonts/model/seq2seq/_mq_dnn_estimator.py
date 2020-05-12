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
import multiprocessing
from typing import List, Optional

# Third-party imports
import numpy as np
import mxnet as mx

# First-party imports
from gluonts.dataset.common import Dataset, ListDataset
from gluonts.dataset.stat import calculate_dataset_statistics
from gluonts.block.decoder import ForkingMLPDecoder
from gluonts.block.encoder import HierarchicalCausalConv1DEncoder, RNNEncoder
from gluonts.block.quantile_output import QuantileOutput
from gluonts.core.component import validated
from gluonts.trainer import Trainer
from gluonts.model.seq2seq._forking_estimator import ForkingSeq2SeqEstimator


class MQCNNEstimator(ForkingSeq2SeqEstimator):
    """
    An :class:`MQDNNEstimator` with a Convolutional Neural Network (CNN) as an
    encoder and a multi-quantile MLP as a decoder. Implements the MQ-CNN Forecaster, proposed in [WTN+17]_.
    """

    @validated()
    def __init__(
        self,
        freq: str,
        prediction_length: int,
        context_length: Optional[int] = None,
        use_feat_dynamic_real: bool = False,
        use_feat_static_cat: bool = False,
        cardinality: List[int] = None,
        embedding_dimension: List[int] = None,
        add_time_feature: bool = False,
        add_age_feature: bool = False,
        enable_decoder_dynamic_feature: bool = True,
        seed: Optional[int] = None,
        decoder_mlp_dim_seq: Optional[List[int]] = None,
        channels_seq: Optional[List[int]] = None,
        dilation_seq: Optional[List[int]] = None,
        kernel_size_seq: Optional[List[int]] = None,
        use_residual: bool = True,
        quantiles: Optional[List[float]] = None,
        trainer: Trainer = Trainer(),
    ) -> None:

        assert (
            prediction_length > 0
        ), f"Invalid prediction length: {prediction_length}."
        assert decoder_mlp_dim_seq is None or all(
            d > 0 for d in decoder_mlp_dim_seq
        ), "Elements of `mlp_hidden_dimension_seq` should be > 0"
        assert channels_seq is None or all(
            d > 0 for d in channels_seq
        ), "Elements of `channels_seq` should be > 0"
        assert dilation_seq is None or all(
            d > 0 for d in dilation_seq
        ), "Elements of `dilation_seq` should be > 0"
        # TODO: add support for kernel size=1
        assert kernel_size_seq is None or all(
            d > 1 for d in kernel_size_seq
        ), "Elements of `kernel_size_seq` should be > 0"
        assert quantiles is None or all(
            0 <= d <= 1 for d in quantiles
        ), "Elements of `quantiles` should be >= 0 and <= 1"

        self.decoder_mlp_dim_seq = (
            decoder_mlp_dim_seq if decoder_mlp_dim_seq is not None else [20]
        )
        self.channels_seq = (
            channels_seq if channels_seq is not None else [30, 30, 30]
        )
        self.dilation_seq = (
            dilation_seq if dilation_seq is not None else [1, 3, 9]
        )
        self.kernel_size_seq = (
            kernel_size_seq if kernel_size_seq is not None else [3, 3, 3]
        )
        self.quantiles = (
            quantiles
            if quantiles is not None
            else [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        )

        assert (
            len(self.channels_seq)
            == len(self.dilation_seq)
            == len(self.kernel_size_seq)
        ), (
            f"mismatch CNN configurations: {len(self.channels_seq)} vs. "
            f"{len(self.dilation_seq)} vs. {len(self.kernel_size_seq)}"
        )

        if seed:
            np.random.seed(seed)
            mx.random.seed(seed)

        # `use_static_feat` and `use_dynamic_feat` always True because network
        # always receives input; either from the input data or constants
        encoder = HierarchicalCausalConv1DEncoder(
            dilation_seq=self.dilation_seq,
            kernel_size_seq=self.kernel_size_seq,
            channels_seq=self.channels_seq,
            use_residual=use_residual,
            use_static_feat=True,
            use_dynamic_feat=True,
            prefix="encoder_",
        )

        decoder = ForkingMLPDecoder(
            dec_len=prediction_length,
            final_dim=self.decoder_mlp_dim_seq[-1],
            hidden_dimension_sequence=self.decoder_mlp_dim_seq[:-1],
            prefix="decoder_",
        )

        quantile_output = QuantileOutput(self.quantiles)

        super().__init__(
            encoder=encoder,
            decoder=decoder,
            quantile_output=quantile_output,
            freq=freq,
            prediction_length=prediction_length,
            context_length=context_length,
            use_feat_dynamic_real=use_feat_dynamic_real,
            use_feat_static_cat=use_feat_static_cat,
            enable_decoder_dynamic_feature=enable_decoder_dynamic_feature,
            cardinality=cardinality,
            embedding_dimension=embedding_dimension,
            add_time_feature=add_time_feature,
            add_age_feature=add_age_feature,
            trainer=trainer,
        )

    @classmethod
    def derive_auto_fields(cls, train_iter):
        stats = calculate_dataset_statistics(train_iter)

        return {
            "use_feat_dynamic_real": stats.num_feat_dynamic_real > 0,
            "use_feat_static_cat": bool(stats.feat_static_cat),
            "cardinality": [len(cats) for cats in stats.feat_static_cat],
        }

    # FIXME: for now we always want the dataset to be cached and utilize multiprocessing.
    # TODO it properly: Enable caching of the dataset in the `_load_datasets` function of the shell,
    #  and pass `num_workers` from train_env in the `run_train_and_test` method to `run_train`,
    #  which in turn has to pass it to train(...)
    def train(
        self,
        training_data: Dataset,
        validation_data: Optional[Dataset] = None,
        num_workers: Optional[int] = None,
        num_prefetch: Optional[int] = None,
        **kwargs,
    ):
        cached_train_data = ListDataset(
            data_iter=list(training_data), freq=self.freq
        )
        cached_validation_data = (
            None
            if validation_data is None
            else ListDataset(data_iter=list(validation_data), freq=self.freq)
        )
        num_workers = (
            num_workers
            if num_workers is not None
            else int(np.ceil(np.sqrt(multiprocessing.cpu_count())))
        )

        return super().train(
            training_data=cached_train_data,
            validation_data=cached_validation_data,
            num_workers=num_workers,
            num_prefetch=num_prefetch,
            **kwargs,
        )


class MQRNNEstimator(ForkingSeq2SeqEstimator):
    """
    An :class:`MQDNNEstimator` with a Recurrent Neural Network (RNN) as an
    encoder and a multi-quantile MLP as a decoder. Implements the MQ-RNN Forecaster, proposed in [WTN+17]_.
    """

    @validated()
    def __init__(
        self,
        prediction_length: int,
        freq: str,
        context_length: Optional[int] = None,
        decoder_mlp_dim_seq: List[int] = None,
        trainer: Trainer = Trainer(),
        quantiles: List[float] = None,
    ) -> None:

        assert (
            prediction_length > 0
        ), f"Invalid prediction length: {prediction_length}."
        assert decoder_mlp_dim_seq is None or all(
            d > 0 for d in decoder_mlp_dim_seq
        ), "Elements of `mlp_hidden_dimension_seq` should be > 0"
        assert quantiles is None or all(
            0 <= d <= 1 for d in quantiles
        ), "Elements of `quantiles` should be >= 0 and <= 1"

        self.decoder_mlp_dim_seq = (
            decoder_mlp_dim_seq if decoder_mlp_dim_seq is not None else [20]
        )
        self.quantiles = (
            quantiles if quantiles is not None else [0.1, 0.5, 0.9]
        )

        # `use_static_feat` and `use_dynamic_feat` always True because network
        # always receives input; either from the input data or constants
        encoder = RNNEncoder(
            mode="gru",
            hidden_size=50,
            num_layers=1,
            bidirectional=True,
            prefix="encoder_",
            use_static_feat=True,
            use_dynamic_feat=True,
        )

        decoder = ForkingMLPDecoder(
            dec_len=prediction_length,
            final_dim=self.decoder_mlp_dim_seq[-1],
            hidden_dimension_sequence=self.decoder_mlp_dim_seq[:-1],
            prefix="decoder_",
        )

        quantile_output = QuantileOutput(self.quantiles)

        super().__init__(
            encoder=encoder,
            decoder=decoder,
            quantile_output=quantile_output,
            freq=freq,
            prediction_length=prediction_length,
            context_length=context_length,
            trainer=trainer,
        )
