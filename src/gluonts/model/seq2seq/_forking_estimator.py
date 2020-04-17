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
from typing import Optional

# First-party imports
from gluonts.block.decoder import Seq2SeqDecoder
from gluonts.block.enc2dec import PassThroughEnc2Dec
from gluonts.block.encoder import Seq2SeqEncoder
from gluonts.block.quantile_output import QuantileOutput
from gluonts.core.component import validated
from gluonts.dataset.field_names import FieldName
from gluonts.model.estimator import GluonEstimator
from gluonts.model.forecast import Quantile
from gluonts.model.predictor import Predictor, RepresentableBlockPredictor
from gluonts.model.forecast_generator import QuantileForecastGenerator
from gluonts.support.util import copy_parameters
from gluonts.trainer import Trainer
from gluonts.time_feature import time_features_from_frequency_str
from gluonts.transform import (
    AddAgeFeature,
    AddTimeFeatures,
    Chain,
    TestSplitSampler,
    Transformation,
    VstackFeatures,
    RenameFields,
    AddConstFeature,
    RemoveFields,
)

# Relative imports
from ._forking_network import (
    ForkingSeq2SeqNetworkBase,
    ForkingSeq2SeqTrainingNetwork,
    ForkingSeq2SeqPredictionNetwork,
)
from ._transform import ForkingSequenceSplitter


class ForkingSeq2SeqEstimator(GluonEstimator):
    r"""
    Sequence-to-Sequence (seq2seq) structure with the so-called
    "Forking Sequence" proposed in [WTN+17]_.

    The basic idea is that, given a sequence :math:`x_1, x_2, \cdots, x_T`,
    with a decoding length :math:`\tau`, we learn a NN that solves the
    following series of seq2seq problems:

    .. math::
       :nowrap:

       \begin{eqnarray}
       x_1                     & \mapsto & x_{2:2+\tau}\\
       x_1, x_2                & \mapsto & x_{3:3+\tau}\\
       x_1, x_2, x_3           & \mapsto & x_{4:4+\tau}\\
                               & \ldots  & \\
       x_1, \ldots, x_{T-\tau} & \mapsto & x_{T-\tau+1:T}
       \end{eqnarray}

    Essentially, this means instead of having one cut in the standard seq2seq,
    one has multiple cuts that progress linearly.

    Parameters
    ----------
    encoder
        seq2seq encoder
    decoder
        seq2seq decoder
    quantile_output
        quantile output
    freq
        frequency of the time series
    prediction_length
        length of the decoding sequence
    context_length
        length of the encoding sequence (prediction_length is used if None)
    trainer
        trainer
    """

    @validated()
    def __init__(
        self,
        encoder: Seq2SeqEncoder,
        decoder: Seq2SeqDecoder,
        quantile_output: QuantileOutput,
        freq: str,
        prediction_length: int,
        use_feat_dynamic_real: bool = False,
        add_time_feature: bool = False,
        add_age_feature: bool = False,
        context_length: Optional[int] = None,
        trainer: Trainer = Trainer(),
    ) -> None:
        super().__init__(trainer=trainer)

        assert (
            context_length is None or context_length > 0
        ), "The value of `context_length` should be > 0"
        assert (
            prediction_length > 0
        ), "The value of `prediction_length` should be > 0"
        # assert (cardinality and use_feat_static_cat) or (
        #     not (cardinality or use_feat_static_cat)
        # ), "You should set `cardinality` if and only if `use_feat_static_cat=True`"
        # assert cardinality is None or all(
        #     [c > 0 for c in cardinality]
        # ), "Elements of `cardinality` should be > 0"
        # assert embedding_dimension is None or all(
        #     [e > 0 for e in embedding_dimension]
        # ), "Elements of `embedding_dimension` should be > 0"

        self.encoder = encoder
        self.decoder = decoder
        self.quantile_output = quantile_output
        self.freq = freq
        self.prediction_length = prediction_length
        self.context_length = (
            context_length
            if context_length is not None
            else self.prediction_length
        )
        self.use_feat_dynamic_real = use_feat_dynamic_real
        self.add_time_feature = add_time_feature
        self.add_age_feature = add_age_feature
        self.use_dynamic_feat = (
            use_feat_dynamic_real or add_age_feature or add_time_feature
        )

        # self.use_feat_static_cat = use_feat_static_cat
        # self.cardinality = (
        #     cardinality if cardinality and use_feat_static_cat else [1]
        # )
        # self.embedding_dimension = (
        #     embedding_dimension
        #     if embedding_dimension is not None
        #     else [min(50, (cat + 1) // 2) for cat in self.cardinality]
        # )

    def create_transformation(self) -> Transformation:
        chain = []
        dynamic_feat_fields = []

        if self.add_time_feature:
            chain.append(
                AddTimeFeatures(
                    start_field=FieldName.START,
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_TIME,
                    time_features=time_features_from_frequency_str(self.freq),
                    pred_length=self.prediction_length,
                ),
            )
            dynamic_feat_fields.append(FieldName.FEAT_TIME)

        if self.add_age_feature:
            chain.append(
                AddAgeFeature(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_AGE,
                    pred_length=self.prediction_length,
                ),
            )
            dynamic_feat_fields.append(FieldName.FEAT_AGE)

        if self.use_feat_dynamic_real:
            dynamic_feat_fields.append(FieldName.FEAT_DYNAMIC_REAL)
        else:
            chain.append(
                RemoveFields(field_names=[FieldName.FEAT_DYNAMIC_REAL])
            )

        # we need to make sure that there is always some dynamic input
        # we will however disregard it in the hybrid forward
        if len(dynamic_feat_fields) == 0:
            chain.append(
                AddConstFeature(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_CONST,
                    pred_length=self.prediction_length,
                ),
            )
            dynamic_feat_fields.append(FieldName.FEAT_CONST)

        # now we map all the dynamic input onto FieldName.FEAT_DYNAMIC_REAL
        # TODO: change the field from FieldName.FEAT_DYNAMIC_REAL to FieldName.FEAT_TIME for consistency with deepAR
        #  or to FieldName.FEAT_DYNAMIC, which would have to be added
        if len(dynamic_feat_fields) > 1:
            chain.append(
                VstackFeatures(
                    output_field=FieldName.FEAT_DYNAMIC_REAL,
                    input_fields=dynamic_feat_fields,
                )
            )
        elif (
            len(dynamic_feat_fields) == 1
            and FieldName.FEAT_DYNAMIC_REAL not in dynamic_feat_fields
        ):
            chain.append(
                RenameFields(
                    {dynamic_feat_fields[0]: FieldName.FEAT_DYNAMIC_REAL}
                )
            )

        chain.append(
            # because of how the forking decoder works, every time step
            # in context is used for splitting, which is why we use the TestSplitSampler
            ForkingSequenceSplitter(
                train_sampler=TestSplitSampler(),
                enc_len=self.context_length,
                dec_len=self.prediction_length,
                encoder_series_fields=[
                    FieldName.FEAT_DYNAMIC_REAL
                ],  # TODO: later add categorical too
            ),
        )

        return Chain(chain)

    def create_training_network(self) -> ForkingSeq2SeqNetworkBase:
        return ForkingSeq2SeqTrainingNetwork(
            encoder=self.encoder,
            enc2dec=PassThroughEnc2Dec(),
            decoder=self.decoder,
            quantile_output=self.quantile_output,
            use_dynamic_feat=self.use_dynamic_feat,
        )

    def create_predictor(
        self,
        transformation: Transformation,
        trained_network: ForkingSeq2SeqNetworkBase,
    ) -> Predictor:
        # todo: this is specific to quantile output
        quantile_strs = [
            Quantile.from_float(quantile).name
            for quantile in self.quantile_output.quantiles
        ]

        prediction_network = ForkingSeq2SeqPredictionNetwork(
            encoder=trained_network.encoder,
            enc2dec=trained_network.enc2dec,
            decoder=trained_network.decoder,
            quantile_output=trained_network.quantile_output,
            use_dynamic_feat=trained_network.use_dynamic_feat,
        )

        copy_parameters(trained_network, prediction_network)

        return RepresentableBlockPredictor(
            input_transform=transformation,
            prediction_net=prediction_network,
            batch_size=self.trainer.batch_size,
            freq=self.freq,
            prediction_length=self.prediction_length,
            ctx=self.trainer.ctx,
            forecast_generator=QuantileForecastGenerator(quantile_strs),
        )
