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
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from gluonts.core.component import validated
from gluonts.dataset.common import Dataset
from gluonts.dataset.field_names import FieldName
from gluonts.model.estimator import Estimator
from gluonts.model.psagan.cnn_encoder._data_CNN import SubSeriesDatasetCNN
from gluonts.model.psagan.cnn_encoder._model import CausalCNNEncoder
from gluonts.model.psagan.cnn_encoder._trainer import Trainer
from gluonts.time_feature import time_features_from_frequency_str
from gluonts.transform import (
    AddObservedValuesIndicator,
    AddTimeFeatures,
    AsNumpyArray,
    Chain,
    RemoveFields,
    SetField,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S]",
)
logger = logging.getLogger(__name__)


class CausalCNNEncoderEstimator(Estimator):
    @validated()
    def __init__(
        self,
        trainer: Trainer,
        freq: str,
        batch_size: int = 24,
        nb_features: int = 7,
        nb_channels: int = 10,
        depth: int = 3,
        reduced_size: int = 10,
        size_embedding: int = 10,
        kernel_size: int = 4,
        subseries_length: int = 20,
        context_length: int = 40,
        max_len: int = 250,
        nb_negative_samples: int = 10,
        use_feat_dynamic_real: bool = True,
        use_feat_static_cat: bool = True,
        use_feat_static_real: bool = False,
        device: str = "cpu",
        scaling: str = "global",
        num_workers: int = 4,
    ):
        super().__init__()
        self.trainer = trainer
        self.batch_size = batch_size
        self.nb_features = nb_features
        self.nb_channels = nb_channels
        self.depth = depth
        self.reduced_size = reduced_size
        self.size_embedding = size_embedding
        self.kernel_size = kernel_size
        self.subseries_length = subseries_length
        self.context_length = context_length
        self.max_len = max_len
        self.nb_negative_samples = nb_negative_samples
        self.use_feat_dynamic_real = use_feat_dynamic_real
        self.use_feat_static_cat = use_feat_static_cat
        self.use_feat_static_real = use_feat_static_real
        self.freq = freq
        assert device == "cpu" or device == "gpu"
        self.device = torch.device("cuda" if device == "gpu" else "cpu")
        self.num_workers = num_workers
        assert (
            scaling == "local" or scaling == "global"
        ), "scaling has to be \
            local or global. If it is local, then the whole time series will be\
            mix-max scaled. If it is local, then each subseries of length \
            target_len will be min-max scaled independenty."
        self.scaling = scaling

    def create_training_network(self):
        net = CausalCNNEncoder(
            in_channels=self.nb_features,
            channels=self.nb_channels,
            depth=self.depth,
            reduced_size=self.reduced_size,
            out_channels=self.size_embedding,
            kernel_size=self.kernel_size,
        ).to(self.device)
        return net

    def create_transformation(self):
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
                    pred_length=self.subseries_length,
                ),
            ]
        )

        return transformation

    def train(
        self,
        training_data: Dataset,
        validation_data: Optional[Dataset] = None,
    ):
        transformation = self.create_transformation()
        dataset = SubSeriesDatasetCNN(
            dataset=training_data,
            transformation=transformation,
            nb_negative_samples=self.nb_negative_samples,
            max_len=self.max_len,
            batch_size=self.batch_size,
            device=self.device,
            scaling=self.scaling,
        )
        dataloader = DataLoader(
            dataset=dataset, batch_size=self.batch_size
        )  # , num_workers=self.num_workers)
        model = self.create_training_network()

        self.trained_model = self.trainer(
            data_loader=dataloader, network=model
        )

        return self.trained_model
