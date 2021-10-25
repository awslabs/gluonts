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
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from gluonts.core.component import validated
from gluonts.dataset.common import Dataset
from gluonts.dataset.field_names import FieldName
from gluonts.model.estimator import Estimator
from gluonts.model.psagan._dataload import Data
from gluonts.model.psagan._model import (
    ProDiscriminator,
    ProGenDiscrim,
    ProGenerator,
    ProGeneratorInference,
)
from gluonts.model.psagan._trainer import Trainer
from gluonts.time_feature import time_features_from_frequency_str
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.transform import (
    AddAgeFeature,
    AddObservedValuesIndicator,
    AddPositionalEncoding,
    AddTimeFeatures,
    AsNumpyArray,
    Chain,
    ExpectedNumInstanceSampler,
    InstanceSplitter,
    RemoveFields,
    SetField,
    TestSplitSampler,
    VstackFeatures,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S]",
)
logger = logging.getLogger(__name__)


class syntheticEstimator(Estimator):
    @validated()
    def __init__(
        self,
        trainer: Trainer,
        freq: str,
        batch_size: int,
        num_batches_per_epoch: int,
        target_len: int,
        nb_features: int,
        ks_conv: int,
        key_features: int,
        value_features: int,
        ks_value: int,
        ks_query: int,
        ks_key: int,
        path_to_pretrain: str = None,
        use_feat_dynamic_real: bool = False,
        use_feat_static_cat: bool = True,
        use_feat_static_real: bool = False,
        num_workers: int = 0,
        device: str = "cpu",
        scaling: str = "local",
        cardinality: Optional[List[int]] = None,
        embedding_dim: int = 10,
        self_attention: bool = True,
        channel_nb: int = 32,
        pos_enc_dimension: int = 10,
        context_length: int = 0,
        exclude_index: list = None,
    ):
        super().__init__()
        self.trainer = trainer
        self.freq = freq
        self.batch_size = batch_size
        self.num_batches_per_epoch = num_batches_per_epoch
        self.target_len = target_len
        self.nb_features = nb_features
        self.ks_conv = ks_conv
        self.key_features = key_features
        self.value_features = value_features
        self.ks_value = ks_value
        self.ks_query = ks_query
        self.ks_key = ks_key
        self.use_feat_dynamic_real = use_feat_dynamic_real
        self.use_feat_static_cat = use_feat_static_cat
        self.use_feat_static_real = use_feat_static_real
        self.num_workers = num_workers
        self.channel_nb = channel_nb
        assert device == "cpu" or device == "gpu"
        self.device = torch.device("cuda" if device == "gpu" else "cpu")
        self.path_to_pretrain = path_to_pretrain
        assert (
            scaling == "local" or scaling == "global" or scaling == "NoScale"
        ), "scaling has to be \
            local or global. If it is local, then the whole time series will be\
            mix-max scaled. If it is local, then each subseries of length \
            target_len will be min-max scaled independenty. If is is NoScale\
            then no scaling is applied to the dataset."
        self.scaling = scaling
        self.cardinality = cardinality
        self.embedding_dim = embedding_dim
        self.self_attention = self_attention
        self.pos_enc_dimension = pos_enc_dimension
        self.context_length = context_length
        assert context_length <= target_len
        if exclude_index is None:
            self.exclude_index = []
        else:
            self.exclude_index = exclude_index

    def create_transformation(
        self, instance_splitter: bool = False, train=True
    ):
        time_features = time_features_from_frequency_str(self.freq)
        remove_field_names = [FieldName.FEAT_DYNAMIC_CAT]
        if not self.use_feat_static_real:
            remove_field_names.append(FieldName.FEAT_STATIC_REAL)
        if not self.use_feat_dynamic_real:
            remove_field_names.append(FieldName.FEAT_DYNAMIC_REAL)
        transformation = Chain(
            [RemoveFields(field_names=remove_field_names)]
            + (
                [SetField(output_field=FieldName.FEAT_STATIC_CAT, value=[0.0])]
                if not self.use_feat_static_cat
                else []
            )
            + (
                [
                    SetField(
                        output_field=FieldName.FEAT_STATIC_REAL, value=[0.0]
                    )
                ]
                if not self.use_feat_static_real
                else []
            )
            + [
                AddObservedValuesIndicator(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.OBSERVED_VALUES,
                ),
                AsNumpyArray(
                    field=FieldName.FEAT_STATIC_CAT,
                    expected_ndim=1,
                    dtype=np.float32,
                ),
                AddTimeFeatures(
                    start_field=FieldName.START,
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_TIME,
                    time_features=time_features,
                    pred_length=self.target_len,
                ),
            ]
            + (
                [
                    AddPositionalEncoding(
                        start_field=FieldName.START,
                        target_field=FieldName.TARGET,
                        output_field=FieldName.POS_ENCODING,
                        pos_enc_dimension=self.pos_enc_dimension,
                    ),
                    VstackFeatures(
                        output_field=FieldName.FEAT_TIME,
                        input_fields=[
                            FieldName.FEAT_TIME,
                            FieldName.POS_ENCODING,
                        ],
                    ),
                ]
                if self.pos_enc_dimension > 0
                else []
            )
            + [
                AddAgeFeature(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_AGE,
                    pred_length=self.target_len,
                    log_scale=True,
                    dtype=np.float32,
                ),
                VstackFeatures(
                    output_field=FieldName.FEAT_TIME,
                    input_fields=[FieldName.FEAT_TIME, FieldName.FEAT_AGE]
                    + (
                        [FieldName.FEAT_DYNAMIC_REAL]
                        if self.use_feat_dynamic_real
                        else []
                    ),
                ),
            ]
            + (
                [
                    InstanceSplitter(
                        target_field=FieldName.TARGET,
                        is_pad_field=FieldName.IS_PAD,
                        start_field=FieldName.START,
                        forecast_start_field=FieldName.FORECAST_START,
                        instance_sampler=ExpectedNumInstanceSampler(
                            num_instances=1,
                            min_future=self.target_len,
                        )
                        if train
                        else TestSplitSampler(
                            min_past=self.target_len,
                        ),
                        past_length=self.target_len,
                        future_length=self.target_len,
                        time_series_fields=[
                            FieldName.FEAT_TIME,
                            FieldName.OBSERVED_VALUES,
                        ],
                    )
                ]
                if instance_splitter
                else []
            ),
        )

        return transformation

    def _str2bool(self, v):
        return not (v.lower() in ("false"))

    def create_training_network(self):
        gen = ProGenerator(
            target_len=self.target_len,
            nb_features=self.nb_features,
            ks_conv=self.ks_conv,
            key_features=self.key_features,
            value_features=self.value_features,
            ks_value=self.ks_value,
            ks_query=self.ks_query,
            ks_key=self.ks_key,
            scaling=self.scaling,
            device=self.device.type,
            cardinality=self.cardinality,
            embedding_dim=self.embedding_dim,
            self_attention=self.self_attention,
            channel_nb=self.channel_nb,
            context_length=self.context_length,
        )
        discrim = ProDiscriminator(
            target_len=self.target_len,
            nb_features=self.nb_features,
            ks_conv=self.ks_conv,
            key_features=self.key_features,
            value_features=self.value_features,
            ks_value=self.ks_value,
            ks_query=self.ks_query,
            ks_key=self.ks_key,
            cardinality=self.cardinality,
            embedding_dim=self.embedding_dim,
            self_attention=self.self_attention,
            channel_nb=self.channel_nb,
        )

        network = ProGenDiscrim(discriminator=discrim, generator=gen)
        if self._str2bool(self.path_to_pretrain):
            network.load_state_dict(
                torch.load(
                    self.path_to_pretrain, map_location=torch.device("cpu")
                )["generator_model_state_dict"]
            )
            logger.info("Pre trained has been loaded")
        else:
            logger.info("No pre trained model has been loaded")

        return network

    def train(
        self, training_data: Dataset, validation_data: Optional[Dataset] = None
    ):
        transformation = self.create_transformation()
        # dataloader = TrainDataLoader(
        #     training_data,
        #     batch_size=self.batch_size,
        #     num_batches_per_epoch=self.num_batches_per_epoch,
        #     transform=transformation,
        #     stack_fn=partial(batchify, device = self.device),
        #     num_workers=self.num_workers
        # )
        ds = Data(
            training_data,
            transformation,
            self.num_batches_per_epoch,
            self.target_len,
            self.batch_size,
            device=self.device,
            scaling=self.scaling,
            context_length=self.context_length,
            exclude_index=self.exclude_index,
        )
        dataloader = DataLoader(ds, batch_size=self.batch_size)
        net = self.create_training_network()
        logger.info(
            f"Batch Size : {self.batch_size},\
            Number of Epochs : {self.trainer.num_epochs},\
            Learning rate Generator : {self.trainer.lr_generator},\
            Learning rate Discriminator : {self.trainer.lr_discriminator},\
            Betas Generator : {self.trainer.betas_generator},\
            Betas Discriminator : {self.trainer.betas_discriminator},\
            Momment Loss : {self.trainer.momment_loss},\
            Loss version : {self.trainer.use_loss},\
            Pre-Trained Generator : {self.path_to_pretrain},\
            Number epoch to fade in layer : {self.trainer.nb_epoch_fade_in_new_layer},\
            Target length : {self.target_len},\
            Kernel size : {self.ks_conv},\
            Key features : {self.key_features},\
            Value features : {self.value_features},\
            Kernel size value : {self.ks_value},\
            Kernel size query : {self.ks_query},\
            Kernel size key : {self.ks_key},\
            Scaling : {self.scaling},\
            Cardinality : {self.cardinality},\
            embedding_dim : {self.embedding_dim}\
            channel nb: {self.channel_nb}\
            positional encoding: {self.pos_enc_dimension}"
        )
        self.trained_net = self.trainer(data_loader=dataloader, network=net)

        predictor = PyTorchPredictor(
            prediction_length=self.target_len,
            input_names=["past_target", "future_time_feat", "feat_static_cat"],
            batch_size=self.batch_size,
            freq=self.freq,
            prediction_net=ProGeneratorInference(self.trained_net.generator),
            input_transform=self.create_transformation(
                instance_splitter=True, train=False
            ),
            device="cpu",
        )

        return predictor
