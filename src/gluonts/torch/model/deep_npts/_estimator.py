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

from copy import deepcopy
from typing import List, Optional, Callable, Union
from functools import partial

import torch

from gluonts.dataset.common import Dataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.loader import DataLoader, TrainDataLoader
from gluonts.itertools import Cached
from gluonts.model.estimator import Estimator
from gluonts.time_feature import time_features_from_frequency_str
from gluonts.torch.batchify import batchify
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.transform import (
    Chain,
    AddAgeFeature,
    AddObservedValuesIndicator,
    AddTimeFeatures,
    AsNumpyArray,
    InstanceSplitter,
    ExpectedNumInstanceSampler,
    RemoveFields,
    SetField,
    TestSplitSampler,
    Transformation,
    TransformedDataset,
    VstackFeatures,
)

from ._network import (
    DeepNPTSNetwork,
    DeepNPTSNetworkDiscrete,
    DeepNPTSMultiStepPredictor,
)
from .scaling import (
    min_max_scaling,
    standard_normal_scaling,
)

LOSS_SCALING_MAP = {
    "min_max_scaling": partial(min_max_scaling, dim=1, keepdim=False),
    "standard_normal_scaling": partial(
        standard_normal_scaling, dim=1, keepdim=False
    ),
}


class DeepNPTSEstimator(Estimator):
    """
    Construct a DeepNPTS estimator. This is a tunable extension of NPTS
    where the sampling probabilities are learned from the data. This is a
    global-model unlike NPTS.

    Currently two variants of the model are implemented:
    (i) `DeepNPTSNetworkDiscrete`: the forecast distribution is a discrete
    distribution similar to NPTS and the forecasts are sampled from the
    observations in the context window.
    (ii) `DeepNPTSNetworkSmooth`: the forecast distribution is a smoothed
    mixture distribution where the components of the mixture are Gaussians
    centered around the observations in the context window. The mixing
    probabilities and the width of the Gaussians are learned. Here the
    forecast can contain values not observed in the context window.

    Parameters
    ----------
    freq
        Frequency of the data to train on and predict
    prediction_length
        Length of the prediction horizon
    context_length
        Number of steps to unroll the RNN for before computing predictions
        (default: None, in which case context_length = prediction_length)
    num_hidden_nodes
        A list containing the number of nodes in each hidden layer
    batch_norm
        Flag to indicate if batch normalization should be applied at every
        layer
    use_feat_static_cat
        Whether to use the ``feat_static_cat`` field from the data
        (default: False)
    num_feat_static_real
        Number of static real features in the data set
    num_feat_dynamic_real
        Number of dynamic features in the data set. These features are added
        to the time series features that are automatically created based on
        the frequency
    cardinality
        Number of values of each categorical feature
        This must be set if ``use_feat_static_cat == True`` (default: None)
    embedding_dimension
        Dimension of the embeddings for categorical features
        (default: [min(50, (cat+1)//2) for cat in cardinality])
    input_scaling
        The scaling to be applied to the target values.
        Available options: "min_max_scaling" and "standard_normal_scaling"
        (default: no scaling)
    dropout_rate
        Dropout regularization parameter (default: no dropout)
    network_type
        The network to be used: either the discrete version
        `DeepNPTSNetworkDiscrete` or the smoothed version
        `DeepNPTSNetworkSmooth` (default: DeepNPTSNetworkDiscrete)
    """

    def __init__(
        self,
        freq: str,
        prediction_length: int,
        context_length: int,
        num_hidden_nodes: Optional[List[int]] = None,
        batch_norm: bool = False,
        use_feat_static_cat: bool = False,
        num_feat_static_real: int = 0,
        num_feat_dynamic_real: int = 0,
        cardinality: Optional[List[int]] = None,
        embedding_dimension: Optional[List[int]] = None,
        input_scaling: Optional[Union[Callable, str]] = None,
        dropout_rate: Optional[float] = None,
        network_type: DeepNPTSNetwork = DeepNPTSNetworkDiscrete,
    ):
        assert (cardinality is not None) == use_feat_static_cat, (
            "You should set `cardinality` if and only if"
            " `use_feat_static_cat=True`"
        )
        assert cardinality is None or all(
            [c > 0 for c in cardinality]
        ), "Elements of `cardinality` should be > 0"
        assert embedding_dimension is None or all(
            [e > 0 for e in embedding_dimension]
        ), "Elements of `embedding_dimension` should be > 0"

        self.freq = freq
        self.prediction_length = prediction_length
        self.context_length = context_length

        if num_hidden_nodes is None:
            self.num_hidden_nodes = [context_length] * 2
        else:
            self.num_hidden_nodes = num_hidden_nodes
        self.use_feat_static_cat = use_feat_static_cat
        self.cardinality = (
            cardinality if cardinality and use_feat_static_cat else [1]
        )
        self.embedding_dimension = (
            embedding_dimension
            if embedding_dimension is not None
            else [min(50, (cat + 1) // 2) for cat in self.cardinality]
        )
        self.num_feat_static_real = num_feat_static_real
        self.num_feat_dynamic_real = num_feat_dynamic_real
        self.features_fields = (
            [FieldName.FEAT_STATIC_CAT]
            + [FieldName.FEAT_STATIC_REAL]
            + [
                "past_" + field
                for field in [
                    FieldName.TARGET,
                    FieldName.OBSERVED_VALUES,
                    FieldName.FEAT_TIME,
                ]
            ]
        )
        self.prediction_features_field = ["future_" + FieldName.FEAT_TIME]
        self.target_field = "future_target"
        self.past_target_field = "past_" + FieldName.TARGET
        self.time_features = time_features_from_frequency_str(self.freq)

        # Note that unlike mxnet, which delays the determination of the the
        # number of input nodes until first forward, pytorch requires the
        # number of input nodes upfront (not only for MLP but also for RNN).
        # That is why counting the number of time features and passing it to
        # the network. The count here includes the user-provided dynamic
        # features as well as age feature (that's why +1).
        self.num_time_features = (
            len(self.time_features) + num_feat_dynamic_real + 1
        )

        if isinstance(input_scaling, str):
            assert input_scaling in [
                "min_max_scaling",
                "standard_normal_scaling",
            ], (
                '`input_scaling` must be one of "min_max_scaling" and'
                f' "standard_normal_scaling", but provided "{input_scaling}".'
            )
        self.input_scaling = input_scaling

        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        self.network_type = network_type

    def input_transform(self) -> Transformation:
        # Note: Any change here should be reflected in the
        # `self.num_time_features` field as well.
        remove_field_names = []
        if self.num_feat_static_real == 0:
            remove_field_names.append(FieldName.FEAT_STATIC_REAL)
        if self.num_feat_dynamic_real == 0:
            remove_field_names.append(FieldName.FEAT_DYNAMIC_REAL)

        return Chain(
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
                if not self.num_feat_static_real > 0
                else []
            )
            + [
                AsNumpyArray(
                    field=FieldName.FEAT_STATIC_CAT,
                    expected_ndim=1,
                ),
                AsNumpyArray(
                    field=FieldName.FEAT_STATIC_REAL,
                    expected_ndim=1,
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
                AddAgeFeature(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_AGE,
                    pred_length=self.prediction_length,
                    log_scale=True,
                ),
                VstackFeatures(
                    output_field=FieldName.FEAT_TIME,
                    input_fields=[FieldName.FEAT_TIME, FieldName.FEAT_AGE]
                    + (
                        [FieldName.FEAT_DYNAMIC_REAL]
                        if self.num_feat_dynamic_real > 0
                        else []
                    ),
                ),
            ]
        )

    def instance_splitter(
        self,
        instance_sampler,
        is_train: bool = True,
    ) -> InstanceSplitter:
        return InstanceSplitter(
            target_field=FieldName.TARGET,
            is_pad_field=FieldName.IS_PAD,
            start_field=FieldName.START,
            forecast_start_field=FieldName.FORECAST_START,
            instance_sampler=instance_sampler,
            past_length=self.context_length,
            future_length=1 if is_train else self.prediction_length,
            time_series_fields=[
                FieldName.OBSERVED_VALUES,
                FieldName.FEAT_TIME,
            ],
        )

    def training_data_loader(
        self,
        training_dataset,
        batch_size: int,
        num_batches_per_epoch: int,
    ) -> DataLoader:
        instance_sampler = ExpectedNumInstanceSampler(
            num_instances=self.prediction_length,
            min_past=self.context_length,
            min_future=1,
        )

        return TrainDataLoader(
            training_dataset,
            batch_size=batch_size,
            stack_fn=batchify,
            transform=self.instance_splitter(instance_sampler, is_train=True),
            num_batches_per_epoch=num_batches_per_epoch,
        )

    def train_model(
        self,
        train_dataset: Dataset,
        epochs: int,
        lr: float = 1e-5,
        batch_size: int = 32,
        num_batches_per_epoch: int = 100,
        cache_data: bool = False,
        loss_scaling: Optional[Union[Callable, str]] = None,
    ) -> DeepNPTSNetwork:
        loss_scaling = (
            LOSS_SCALING_MAP[loss_scaling]
            if isinstance(loss_scaling, str)
            else loss_scaling
        )

        transformed_dataset = TransformedDataset(
            train_dataset, self.input_transform()
        )

        data_loader = self.training_data_loader(
            transformed_dataset
            if not cache_data
            else Cached(transformed_dataset),
            batch_size=batch_size,
            num_batches_per_epoch=num_batches_per_epoch,
        )

        net = self.network_type(
            context_length=self.context_length,
            num_hidden_nodes=self.num_hidden_nodes,
            cardinality=self.cardinality,
            embedding_dimension=self.embedding_dimension,
            num_time_features=self.num_time_features,
            input_scaling=self.input_scaling,
            dropout_rate=self.dropout_rate,
            batch_norm=self.batch_norm,
        )

        optimizer = torch.optim.Adam(net.parameters(), lr=lr)

        best_loss = float("inf")
        for epoch_num in range(epochs):
            sum_epoch_loss = 0.0
            for batch_no, batch in enumerate(data_loader, start=1):
                x = {k: batch[k] for k in self.features_fields}
                y = batch[self.target_field]

                predicted_distribution = net(**x)
                scale = (
                    loss_scaling(x[self.past_target_field])[1]
                    if loss_scaling
                    else 1
                )
                loss = (-predicted_distribution.log_prob(y) / scale).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                sum_epoch_loss += loss.detach().numpy().item()

            if sum_epoch_loss < best_loss:
                best_loss = sum_epoch_loss
                best_net = deepcopy(net)

            print(
                f"Loss for epoch {epoch_num}: "
                f"{sum_epoch_loss / num_batches_per_epoch}"
            )

        print(f"Best loss: {best_loss / num_batches_per_epoch}")

        return best_net

    def get_predictor(
        self, net: torch.nn.Module, batch_size: int, device=torch.device("cpu")
    ) -> PyTorchPredictor:
        pred_net_multi_step = DeepNPTSMultiStepPredictor(
            net=net, prediction_length=self.prediction_length
        )

        return PyTorchPredictor(
            prediction_net=pred_net_multi_step,
            prediction_length=self.prediction_length,
            input_names=self.features_fields + self.prediction_features_field,
            batch_size=batch_size,
            input_transform=self.input_transform()
            + self.instance_splitter(TestSplitSampler(), is_train=False),
            device=device,
        )

    def train(
        self,
        train_dataset: Dataset,
        validation_dataset: Optional[Dataset] = None,
        epochs: int = 100,
        lr: float = 1e-5,
        batch_size: int = 32,
        num_batches_per_epoch: int = 100,
        cache_data: bool = False,
        loss_scaling: Optional[Callable] = None,
    ) -> PyTorchPredictor:
        pred_net = self.train_model(
            train_dataset=train_dataset,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            num_batches_per_epoch=num_batches_per_epoch,
            cache_data=cache_data,
            loss_scaling=loss_scaling,
        )

        return self.get_predictor(pred_net, batch_size)
