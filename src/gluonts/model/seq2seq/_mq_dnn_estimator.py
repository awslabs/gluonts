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
from typing import List, Optional

# First-party imports
from gluonts.dataset.stat import calculate_dataset_statistics
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.block.decoder import ForkingMLPDecoder
from gluonts.block.encoder import (
    HierarchicalCausalConv1DEncoder,
    RNNEncoder,
    Seq2SeqEncoder,
)
from gluonts.block.quantile_output import QuantileOutput
from gluonts.core.component import validated
from gluonts.trainer import Trainer

# Relative imports
from gluonts.model.seq2seq._forking_estimator import ForkingSeq2SeqEstimator
from gluonts.evaluation import Evaluator
import numpy as np
import mxnet as mx

# TODO: in general, it seems unnecessary to put the MQCNN and MQRNN into Seq2Seq since their commonality in code with
#  the rest is just the abstract classes Seq2SeqDecoder and Se2SeqEncoder,
#  and the Estimator is not based on Seq2SeqEstimator!


# TODO: THIS CLASS SHOULD NOT EXIST, the decoder
#  can be defined in each current subclass
class MQDNNEstimator(ForkingSeq2SeqEstimator):
    """
    Intermediate base class for a Multi-horizon Quantile Deep Neural Network
    (MQ-DNN), [WTN+17]_. The class fixes the decoder is a multi-quantile MLP.
    Subclasses fix the encoder to be either a Convolutional Neural Network
    (MQ-CNN) or a Recurrent Neural Network (MQ-RNN).
    """

    @validated()
    def __init__(
        self,
        encoder: Seq2SeqEncoder,
        context_length: Optional[int],
        prediction_length: int,
        freq: str,
        use_dynamic_feat: bool = False,
        add_time_feature: bool = False,
        add_age_feature: bool = False,
        decoder_mlp_dim_seq: List[int] = [20],
        quantiles: List[float] = list(),
        trainer: Trainer = Trainer(),
    ) -> None:
        context_length = (
            prediction_length if context_length is None else context_length
        )
        assert all(
            [d > 0 for d in decoder_mlp_dim_seq]
        ), "Elements of `mlp_hidden_dimension_seq` should be > 0"

        decoder = ForkingMLPDecoder(
            dec_len=prediction_length,
            final_dim=decoder_mlp_dim_seq[-1],
            hidden_dimension_sequence=decoder_mlp_dim_seq[:-1],
            prefix="decoder_",
        )

        quantile_output = QuantileOutput(quantiles)

        super().__init__(
            encoder=encoder,
            decoder=decoder,
            quantile_output=quantile_output,
            freq=freq,
            use_dynamic_feat=use_dynamic_feat,
            add_age_feature=add_age_feature,
            add_time_feature=add_time_feature,
            prediction_length=prediction_length,
            context_length=context_length,
            trainer=trainer,
        )


# TODO: integrate MQDNN, change arguments to non mutable
class MQCNNEstimator(MQDNNEstimator):
    """
    An :class:`MQDNNEstimator` with a Convolutional Neural Network (CNN) as an
    encoder. Implements the MQ-CNN Forecaster, proposed in [WTN+17]_.
    """

    @validated()
    def __init__(
        self,
        prediction_length: int,
        freq: str,
        context_length: Optional[int] = None,
        use_feat_dynamic_real: bool = False,
        use_feat_static_cat: bool = False,
        cardinality: Optional[List[int]] = None,
        add_time_feature: bool = False,
        add_age_feature: bool = False,
        seed: Optional[int] = None,
        decoder_mlp_dim_seq: List[int] = [20],
        channels_seq: List[int] = [30, 30, 30],
        dilation_seq: List[int] = [1, 3, 9],
        kernel_size_seq: List[int] = [3, 3, 3],
        use_residual: bool = True,
        quantiles: List[float] = list(
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        ),
        trainer: Trainer = Trainer(),
    ) -> None:

        use_dynamic_feat_cnn = False
        if use_feat_dynamic_real or add_age_feature or add_time_feature:
            use_dynamic_feat_cnn = True

        use_static_feat_cnn = False
        if use_feat_static_cat or use_feat_static_cat:
            use_static_feat_cnn = True

        if seed:
            np.random.seed(seed)
            mx.random.seed(seed)

        assert (
            len(channels_seq) == len(dilation_seq) == len(kernel_size_seq)
        ), (
            f"mismatch CNN configurations: {len(channels_seq)} vs. "
            f"{len(dilation_seq)} vs. {len(kernel_size_seq)}"
        )

        # TODO: figure out whether this needs any additional modification; doesn't seems o
        encoder = HierarchicalCausalConv1DEncoder(
            dilation_seq=dilation_seq,
            kernel_size_seq=kernel_size_seq,
            channels_seq=channels_seq,
            use_residual=use_residual,
            use_dynamic_feat=use_dynamic_feat_cnn,
            prefix="encoder_",
        )

        super().__init__(
            encoder=encoder,
            use_dynamic_feat=use_feat_dynamic_real,  # TODO: make use_dynamic_feat this more specific
            add_time_feature=add_time_feature,
            add_age_feature=add_age_feature,
            decoder_mlp_dim_seq=decoder_mlp_dim_seq,
            freq=freq,
            prediction_length=prediction_length,
            trainer=trainer,
            context_length=context_length,
            quantiles=quantiles,
        )

    @classmethod
    def derive_auto_fields(cls, train_iter):
        stats = calculate_dataset_statistics(train_iter)

        return {
            "use_feat_dynamic_real": stats.num_feat_dynamic_real > 0,
            # "use_feat_static_cat": bool(stats.feat_static_cat),
            # "cardinality": [len(cats) for cats in stats.feat_static_cat],
        }


# TODO: integrate MQDNN, change arguments to non mutable
class MQRNNEstimator(MQDNNEstimator):
    """
    An :class:`MQDNNEstimator` with a Recurrent Neural Network (RNN) as an
    encoder. Implements the MQ-RNN Forecaster, proposed in [WTN+17]_.
    """

    @validated()
    def __init__(
        self,
        prediction_length: int,
        freq: str,
        context_length: Optional[int] = None,
        decoder_mlp_dim_seq: List[int] = [20],
        trainer: Trainer = Trainer(),
        quantiles: List[float] = list([0.1, 0.5, 0.9]),
    ) -> None:
        encoder = RNNEncoder(
            mode="gru",
            hidden_size=50,
            num_layers=1,
            bidirectional=True,
            prefix="encoder_",
        )
        super().__init__(
            encoder=encoder,
            decoder_mlp_dim_seq=decoder_mlp_dim_seq,
            freq=freq,
            prediction_length=prediction_length,
            trainer=trainer,
            context_length=context_length,
            quantiles=quantiles,
        )


# TODO: REMOVE THIS
if __name__ == "__main__":
    from gluonts.dataset.repository.datasets import (
        get_dataset,
        dataset_recipes,
    )

    print(f"datasets available: {dataset_recipes.keys()}")

    # we pick m4_hourly as it only contains a few hundred time series
    dataset = get_dataset("m4_hourly", regenerate=False)

    metrics = []

    for _ in range(1):
        estimator = MQCNNEstimator.from_inputs(
            dataset.train,
            # add_time_feature=True,
            # add_age_feature=True,
            prediction_length=dataset.metadata.prediction_length,
            seed=42,
            freq=dataset.metadata.freq,
            quantiles=[0.5],
            trainer=Trainer(
                epochs=1, num_batches_per_epoch=10, hybridize=True
            ),
        )

        predictor = estimator.train(dataset.train)

        assert dataset.test is not None

        forecast_it, ts_it = make_evaluation_predictions(
            dataset.test, predictor=predictor, num_samples=100
        )

        agg_metrics, item_metrics = Evaluator()(
            ts_it, forecast_it, num_series=len(list(dataset.test))
        )

        metrics.append(agg_metrics["wQuantileLoss[0.5]"])

    print(metrics)
