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

from typing import Any, Optional, List, Dict, Iterable

import numpy as np
import torch

from gluonts.core.component import validated
from gluonts.dataset.common import Dataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.loader import as_stacked_batches
from gluonts.itertools import Cyclic
from gluonts.torch.model.estimator import PyTorchLightningEstimator
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.time_feature import time_features_from_frequency_str
from gluonts.torch.distributions import Output, QuantileOutput

from gluonts.transform import (
    Chain,
    RemoveFields,
    Transformation,
    AsNumpyArray,
    VstackFeatures,
    AddConstFeature,
    AddAgeFeature,
    AddTimeFeatures,
    AddObservedValuesIndicator,
    RenameFields,
    SetField,
    ExpectedNumInstanceSampler,
    TestSplitSampler,
    ValidationSplitSampler,
)
from gluonts.transform.sampler import InstanceSampler
from gluonts.transform.split import ForkingSequenceSplitter

from .lightning_module import MQCNNLightningModule

PREDICTION_INPUT_NAMES = [
    "past_target",
    "past_feat_dynamic",
    "future_feat_dynamic",
    "feat_static_real",
    "feat_static_cat",
    "past_observed_values",
    "past_feat_dynamic_cat",
    "future_feat_dynamic_cat",
]

TRAINING_INPUT_NAMES = PREDICTION_INPUT_NAMES + [
    "future_target",
    "future_observed_values",
]


class MQCNNEstimator(PyTorchLightningEstimator):
    """
    Args:

    freq (str):
        Time granularity of the data.
    prediction_length (int):
        Length of the prediction, also known as 'horizon'.
    context_length (int, optional):
        Number of time units that condition the predictions, also known as 'lookback period'.
        Defaults to `4 * prediction_length`.
    num_forking (int, optional):
        Decides how much forking to do in the decoder.
        (default: context_length if None)
        Defaults to None.
    quantiles (List[float], optional):
        The list of quantiles that will be optimized for, and predicted by, the model.
        (default: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] if None)
        Defaults to None.
    use_past_feat_dynamic_real (bool, optional):
        Whether to use the ``past_feat_dynamic_real`` field from the data.
        Defaults to False.
    use_feat_dynamic_real (bool, optional):
        Whether to use the ``feat_dynamic_real`` field from the data.
        Defaults to False.
    use_feat_dynamic_cat (bool, optional):
        Whether to use the ``feat_dynamic_cat`` field from the data.
        Defaults to False.
    use_feat_static_real (bool, optional):
        Whether to use the ``feat_static_real`` field from the data.
        Defaults to False.
    use_feat_static_cat (bool, optional):
        Whether to use the ``feat_static_cat`` field from the data.
        Defaults to False.
    add_time_feature (bool, optional):
        Adds a set of time features.
        Defaults to True.
    add_age_feature (bool, optional):
        Adds an age feature.
        The age feature starts with a small value at the start of the time series and grows over time.
        Defaults to False.
    enable_encoder_dynamic_feature (bool, optional):
        Whether the encoder should be provided with the dynamic features (``age``, ``time``
        and ``feat_dynamic_real/cat`` if enabled respectively)
        Defaults to True.
    enable_decoder_dynamic_feature (bool, optional):
        Whether the decoder should be provided with the dynamic features (``age``, ``time``
        and ``feat_dynamic_real/cat`` if enabled respectively).
        Defaults to True.
    feat_dynamic_real_dim (int, optional):
        Dimension of real dynamic features.
        Defaults to None.
    past_feat_dynamic_real_dim (int, optional):
        Dimension of past real dynamic features
        Defaults to None.
    cardinality_dynamic (List[int], optional):
        Number of values of each dynamic categorical feature.
        This must be set if ``use_feat_dynamic_cat == True``
        Defaults to None.
    embedding_dimension_dynamic (List[int], optional):
        Dimension of the embeddings for dynamic categorical features.
        (default: [cat for cat in cardinality_dinamic] if None)
        Defaults to None.
    feat_static_real_dim (int, optional):
        Dimension of real static features.
        Defaults to None.
    cardinality_static (List[int], optional):
        Number of values of each static categorical feature.
        This must be set if ``use_feat_static_cat == True``
        Defaults to None.
    embedding_dimension_static (List[int], optional):
        Dimension of the embeddings for categorical features.
        (default: [min(50, (cat+1)//2) for cat in cardinality_static] if None)
        Defaults to None.
    scaling (bool, optional):
        Whether to automatically scale the target values.
        Defaults to None.
    scaling_decoder_dynamic_feature (bool, optional):
        Whether to automatically scale the dynamic features for the decoder.
        Defaults to False.
    joint_embedding_dimension (int, optional):
        Dimension of the joint embedding for all static features (real and categorical) as the end of the encoder
        (default: if None, channels_seq[-1] * sqrt(feat_static_dim)),
        where feat_static_dim is appx sum(embedding_dimension_static))
        Defaults to None.
    time_features (list, optional):
        List of time features, from :py:mod:`gluonts.time_feature`, to use as
        inputs to the model in addition to the provided data.
        Defaults to None, in which case these are automatically determined based
        on freq.
    encoder_mlp_dim_seq (List[int], optional):
        The dimensionalities of the MLP layers of the encoder for static features (default: [] if None)
        Defaults to None.
    decoder_mlp_dim_seq (List[int], optional):
        The dimensionalities of the layers of the local MLP decoder. (default: [30] if None)
        Defaults to None.
    decoder_hidden_dim (int, optional):
        Hidden dimension of the decoder used to produce horizon agnostic and horizon specific encodings of the input.
        (default: 30 if None)
        Defaults to None.
    decoder_future_embedding_dim (int, optional):
        Size of the embeddings used to globally encode future dynamic features.
        (default: 50 if None)
        Defaults to None.
    channels_seq (List[int], optional):
        The number of channels (i.e. filters or convolutions) for each layer of the HierarchicalCausalConv1DEncoder.
        More channels usually correspond to better performance and larger network size.
        (default: [30, 30, 30] if None)
        Defaults to None.
    dilation_seq (List[int], optional):
        The dilation of the convolutions in each layer of the HierarchicalCausalConv1DEncoder.
        Greater numbers correspond to a greater receptive field of the network, which is usually
        better with longer context_length. (Same length as channels_seq) (default: [1, 3, 5] if None)
        Defaults to None.
    kernel_size_seq (List[int], optional):
        The kernel sizes (i.e. window size) of the convolutions in each layer of the HierarchicalCausalConv1DEncoder.
        (Same length as channels_seq) (default: [7, 3, 3] if None)
        Defaults to None.
    use_residual (bool, optional):
        Whether the hierarchical encoder should additionally pass the unaltered
        past target to the decoder.
        Defaults to True.
    batch_size (int, optional):
        The size of the batches to be used training and prediction.
        Defaults to 32.
    val_batch_size(int, optional):
        batch size for validation.
        If None, will use the same batch size
        Defaults to None.
    lr (float, optional)
        Learning rate, by default 1e-3
    learning_rate_decay_factor (float, optional):
        Learning rate decay factor, by default 0.1.
    minimum_learning_rate (float, optional):
        Minimum learning rate, by default 1e-6.
    clip_gradient (float, optional):
        Clip gradient level, by default 10.0.
    weight_decay (float, optional)
        Weight decay, by default 1e-8
    patience (int, optional):
        Patience applied to learning rate scheduling, by deafult 10.
    num_batches_per_epoch (int, optional):
        Number of batches to be processed in each training epoch,
        by default 50
    trainer_kwargs (Dict, optional)
        Additional arguments to provide to ``pl.Trainer`` for construction,
        by default None
    train_sampler (InstanceSampler, optional):
        Controls the sampling of windows during training.
        Defaults to None.
    validation_sampler (InstanceSampler, optional):
        Controls the sampling of windows during validation.
        Defaults to None.
    """

    @validated()
    def __init__(
        self,
        freq: str,
        prediction_length: int,
        context_length: Optional[int] = None,
        num_forking: Optional[int] = None,
        quantiles: Optional[List[float]] = None,
        distr_output: Optional[Output] = None,
        use_past_feat_dynamic_real: bool = False,
        use_feat_dynamic_real: bool = False,
        use_feat_dynamic_cat: bool = False,
        use_feat_static_real: bool = False,
        use_feat_static_cat: bool = False,
        add_time_feature: bool = True,
        add_age_feature: bool = False,
        enable_encoder_dynamic_feature: bool = True,
        enable_decoder_dynamic_feature: bool = True,
        feat_dynamic_real_dim: Optional[int] = None,
        past_feat_dynamic_real_dim: Optional[int] = None,
        cardinality_dynamic: Optional[List[int]] = None,
        embedding_dimension_dynamic: Optional[List[int]] = None,
        feat_static_real_dim: Optional[int] = None,
        cardinality_static: Optional[List[int]] = None,
        embedding_dimension_static: Optional[List[int]] = None,
        scaling: Optional[bool] = None,
        scaling_decoder_dynamic_feature: bool = False,
        joint_embedding_dimension: Optional[int] = None,
        time_features: Optional[list] = None,
        encoder_mlp_dim_seq: Optional[List[int]] = None,
        decoder_mlp_dim_seq: Optional[List[int]] = None,
        decoder_hidden_dim: Optional[int] = None,
        decoder_future_embedding_dim: Optional[int] = None,
        channels_seq: Optional[List[int]] = None,
        dilation_seq: Optional[List[int]] = None,
        kernel_size_seq: Optional[List[int]] = None,
        use_residual: bool = True,
        batch_size: int = 32,
        val_batch_size: Optional[int] = None,
        lr: float = 1e-3,
        learning_rate_decay_factor: float = 0.1,
        minimum_learning_rate: float = 1e-6,
        clip_gradient: float = 10.0,
        weight_decay: float = 1e-8,
        patience: int = 10,
        num_batches_per_epoch: int = 50,
        trainer_kwargs: Dict[str, Any] = None,
        train_sampler: Optional[InstanceSampler] = None,
        validation_sampler: Optional[InstanceSampler] = None,
    ) -> None:

        torch.set_default_tensor_type(torch.FloatTensor)

        default_trainer_kwargs = {
            "max_epochs": 100,
            "gradient_clip_val": clip_gradient,
        }
        if trainer_kwargs is not None:
            default_trainer_kwargs.update(trainer_kwargs)
        super().__init__(trainer_kwargs=default_trainer_kwargs)

        self.freq = freq
        self.prediction_length = prediction_length
        self.context_length = self.context_length = (
            context_length
            if context_length is not None
            else 4 * self.prediction_length
        )
        self.num_forking = (
            min(num_forking, self.context_length)
            if num_forking is not None
            else self.context_length
        )

        # Model architecture
        if distr_output is not None and quantiles is not None:
            raise ValueError(
                "Only one of `distr_output` and `quantiles` must be specified"
            )
        elif distr_output is not None:
            self.distr_output = distr_output
        else:
            if quantiles is None:
                quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            self.distr_output = QuantileOutput(quantiles=quantiles)

        if time_features is None:
            time_features = time_features_from_frequency_str(self.freq)
        self.time_features = time_features

        self.use_past_feat_dynamic_real = use_past_feat_dynamic_real
        self.use_feat_dynamic_real = use_feat_dynamic_real
        self.use_feat_dynamic_cat = use_feat_dynamic_cat
        self.use_feat_static_real = use_feat_static_real
        self.use_feat_static_cat = use_feat_static_cat

        self.add_time_feature = add_time_feature
        self.add_age_feature = add_age_feature
        self.use_dynamic_feat = (
            use_feat_dynamic_real
            or add_age_feature
            or add_time_feature
            or use_feat_dynamic_cat
        )

        self.enable_encoder_dynamic_feature = enable_encoder_dynamic_feature
        self.enable_decoder_dynamic_feature = enable_decoder_dynamic_feature

        self.scaling = scaling if scaling is not None else False
        self.scaling_decoder_dynamic_feature = scaling_decoder_dynamic_feature

        self.train_sampler = train_sampler or ExpectedNumInstanceSampler(
            num_instances=1.0, min_future=prediction_length
        )
        self.validation_sampler = validation_sampler or ValidationSplitSampler(
            min_future=prediction_length
        )

        self.enc_cnn_init_dim = 3  # target, observed, const
        self.dec_future_init_dim = 1  # observed
        if add_time_feature:
            self.enc_cnn_init_dim += 1
            self.dec_future_init_dim += 1
        if add_age_feature:
            self.enc_cnn_init_dim += 1
            self.dec_future_init_dim += 1

        # Training procedure
        self.lr = lr
        self.batch_size = batch_size
        self.val_batch_size = (
            val_batch_size if val_batch_size is not None else batch_size
        )
        self.learning_rate_decay_factor = learning_rate_decay_factor
        self.minimum_learning_rate = minimum_learning_rate
        self.weight_decay = weight_decay
        self.patience = patience
        self.num_batches_per_epoch = num_batches_per_epoch

        self.encoder_mlp_dim_seq = (
            encoder_mlp_dim_seq if encoder_mlp_dim_seq is not None else []
        )
        self.decoder_mlp_dim_seq = (
            decoder_mlp_dim_seq if decoder_mlp_dim_seq is not None else [30]
        )
        self.decoder_hidden_dim = (
            decoder_hidden_dim if decoder_hidden_dim is not None else 30
        )
        self.decoder_future_embedding_dim = (
            decoder_future_embedding_dim
            if decoder_future_embedding_dim is not None
            else 50
        )
        self.channels_seq = (
            channels_seq if channels_seq is not None else [30, 30, 30]
        )
        self.dilation_seq = (
            dilation_seq if dilation_seq is not None else [1, 3, 9]
        )
        self.kernel_size_seq = (
            kernel_size_seq if kernel_size_seq is not None else [7, 3, 3]
        )

        assert (
            len(self.channels_seq)
            == len(self.dilation_seq)
            == len(self.kernel_size_seq)
        ), (
            f"mismatch CNN configurations: {len(self.channels_seq)} vs. "
            f"{len(self.dilation_seq)} vs. {len(self.kernel_size_seq)}"
        )

        self.use_residual = use_residual

        if self.use_feat_dynamic_cat:
            self.cardinality_dynamic = cardinality_dynamic
            self.embedding_dimension_dynamic = (
                embedding_dimension_dynamic
                if embedding_dimension_dynamic is not None
                else [cat for cat in cardinality_dynamic]
            )

            self.enc_cnn_init_dim += sum(self.embedding_dimension_dynamic)
            self.dec_future_init_dim += sum(self.embedding_dimension_dynamic)
        else:
            self.cardinality_dynamic = [0]
            self.embedding_dimension_dynamic = [0]

        if self.use_past_feat_dynamic_real:
            assert (
                past_feat_dynamic_real_dim is not None
            ), "past_feat_dynamic_real should be provided"
            self.enc_cnn_init_dim += past_feat_dynamic_real_dim
            self.past_feat_dynamic_real_dim = past_feat_dynamic_real_dim
        else:
            self.past_feat_dynamic_real_dim = 0

        if self.use_feat_dynamic_real:
            assert (
                feat_dynamic_real_dim is not None
            ), "dim_feat_dynamic_real should be provided"
            self.enc_cnn_init_dim += feat_dynamic_real_dim
            self.dec_future_init_dim += feat_dynamic_real_dim
            self.feat_dynamic_real_dim = feat_dynamic_real_dim
        else:
            self.feat_dynamic_real_dim = 0

        self.enc_mlp_init_dim = 1  # start with 1 because of scaler
        if self.use_feat_static_cat:
            self.cardinality_static = cardinality_static
            self.embedding_dimension_static = (
                embedding_dimension_static
                if embedding_dimension_static is not None
                else [min(50, (cat + 1) // 2) for cat in cardinality_static]
            )
            self.enc_mlp_init_dim += sum(self.embedding_dimension_static)
        else:
            self.cardinality_static = [0]
            self.embedding_dimension_static = [0]

        self.joint_embedding_dimension = joint_embedding_dimension
        if self.joint_embedding_dimension is None:
            feat_static_dim = sum(self.embedding_dimension_static)
            self.joint_embedding_dimension = int(
                self.channels_seq[-1] * max(np.sqrt(feat_static_dim), 1)
            )

        if self.use_feat_static_real:
            assert (
                feat_static_real_dim is not None
            ), "feat_static_real should be provided"
            self.enc_mlp_init_dim += feat_static_real_dim
            self.feat_static_real_dim = feat_static_real_dim
        else:
            self.feat_static_real_dim = 0

    def create_transformation(self) -> Chain:
        """Creates transformation to be applied to input dataset

        Returns:
            Chain:
                transformation chain to be applied to the input data
        """

        dynamic_feat_fields = []
        remove_field_names = []

        if not self.use_past_feat_dynamic_real:
            remove_field_names.append(FieldName.PAST_FEAT_DYNAMIC_REAL)
        if not self.use_feat_dynamic_real:
            remove_field_names.append(FieldName.FEAT_DYNAMIC_REAL)
        if not self.use_feat_dynamic_cat:
            remove_field_names.append(FieldName.FEAT_DYNAMIC_CAT)
        if not self.use_feat_static_real:
            remove_field_names.append(FieldName.FEAT_STATIC_REAL)
        if not self.use_feat_static_cat:
            remove_field_names.append(FieldName.FEAT_STATIC_CAT)

        transforms = [
            RemoveFields(field_names=remove_field_names),
            AsNumpyArray(field=FieldName.TARGET, expected_ndim=1),
            AddObservedValuesIndicator(
                target_field=FieldName.TARGET,
                output_field=FieldName.OBSERVED_VALUES,
            ),
        ]

        if len(self.time_features) > 0:
            transforms.append(
                AddTimeFeatures(
                    start_field=FieldName.START,
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_TIME,
                    time_features=self.time_features,
                    pred_length=self.prediction_length,
                )
            )
            dynamic_feat_fields.append(FieldName.FEAT_TIME)

        if self.add_age_feature:
            transforms.append(
                AddAgeFeature(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_AGE,
                    pred_length=self.prediction_length,
                )
            )
            dynamic_feat_fields.append(FieldName.FEAT_AGE)

        if self.use_feat_dynamic_real:
            # Backwards compatibility:
            transforms.append(
                RenameFields({"dynamic_feat": FieldName.FEAT_DYNAMIC_REAL})
            )
            dynamic_feat_fields.append(FieldName.FEAT_DYNAMIC_REAL)

        # we need to make sure that there is always some dynamic input
        # we will however disregard it in the hybrid forward.
        # the time feature is empty for yearly freq so also adding a dummy feature
        # in the case that the time feature is the only one on
        if len(dynamic_feat_fields) == 0 or (
            not self.add_age_feature
            and not self.add_time_feature
            and not self.use_feat_dynamic_real
        ):
            transforms.append(
                AddConstFeature(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_CONST,
                    pred_length=self.prediction_length,
                    const=0.0,
                )
            )
            dynamic_feat_fields.append(FieldName.FEAT_CONST)

        # now we map all the dynamic input of length context_length + prediction_length onto FieldName.FEAT_DYNAMIC
        # we exclude past_feat_dynamic_real since its length is only context_length
        if len(dynamic_feat_fields) > 1:
            transforms.append(
                VstackFeatures(
                    output_field=FieldName.FEAT_DYNAMIC,
                    input_fields=dynamic_feat_fields,
                )
            )
        elif len(dynamic_feat_fields) == 1:
            transforms.append(
                RenameFields({dynamic_feat_fields[0]: FieldName.FEAT_DYNAMIC})
            )

        if not self.use_feat_dynamic_cat:
            transforms.append(
                AddConstFeature(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_DYNAMIC_CAT,
                    pred_length=self.prediction_length,
                    const=0,
                )
            )

        if not self.use_feat_static_cat:
            transforms.append(
                SetField(output_field=FieldName.FEAT_STATIC_CAT, value=[0])
            )
        transforms.append(
            AsNumpyArray(
                field=FieldName.FEAT_STATIC_CAT,
                expected_ndim=1,
                dtype=np.int64,
            )
        )

        if not self.use_feat_static_real:
            transforms.append(
                SetField(output_field=FieldName.FEAT_STATIC_REAL, value=[0.0])
            )
        transforms.append(
            AsNumpyArray(field=FieldName.FEAT_STATIC_REAL, expected_ndim=1)
        )

        return Chain(transforms)

    def _create_instance_splitter(self, mode: str) -> Chain:
        """Creates instance splitter to be applied to the dataset

        Args:
            mode (str): `training`, `validation` or `test`

        Returns:
            Chain:
                transformation chain to split input data along the time
                dimension before processing
        """

        assert mode in ["training", "validation", "test"]

        instance_sampler = {
            "training": self.train_sampler,
            "validation": self.validation_sampler,
            "test": TestSplitSampler(),
        }[mode]

        chain = []

        encoder_series_fields = [
            FieldName.OBSERVED_VALUES,
            FieldName.FEAT_DYNAMIC,
            FieldName.FEAT_DYNAMIC_CAT,
        ] + (
            [FieldName.PAST_FEAT_DYNAMIC_REAL]
            if self.use_past_feat_dynamic_real
            else []
        )

        encoder_disabled_fields = (
            [FieldName.FEAT_DYNAMIC, FieldName.FEAT_DYNAMIC_CAT]
            if not self.enable_encoder_dynamic_feature
            else []
        ) + (
            [FieldName.PAST_FEAT_DYNAMIC_REAL]
            if not self.enable_encoder_dynamic_feature
            and self.use_past_feat_dynamic_real
            else []
        )

        decoder_series_fields = (
            [
                FieldName.FEAT_DYNAMIC,
                FieldName.FEAT_DYNAMIC_CAT,
            ]
            + ([FieldName.OBSERVED_VALUES] if mode != "test" else [])
        )

        decoder_disabled_fields = (
            [FieldName.FEAT_DYNAMIC, FieldName.FEAT_DYNAMIC_CAT]
            if not self.enable_decoder_dynamic_feature
            else []
        )

        chain.append(
            # because of how the forking decoder works, every time step
            # in context is used for splitting, which is why we use the TestSplitSampler
            ForkingSequenceSplitter(
                instance_sampler=instance_sampler,
                enc_len=self.context_length,
                dec_len=self.prediction_length,
                num_forking=self.num_forking,
                target_field=FieldName.TARGET,
                encoder_series_fields=encoder_series_fields,
                encoder_disabled_fields=encoder_disabled_fields,
                decoder_series_fields=decoder_series_fields,
                decoder_disabled_fields=decoder_disabled_fields,
                prediction_time_decoder_exclude=[FieldName.OBSERVED_VALUES],
                is_pad_out=FieldName.IS_PAD,
                start_input_field=FieldName.START,
            )
        )

        # past_feat_dynamic features generated above in ForkingSequenceSplitter from those under feat_dynamic - we need
        # to stack with the other short related time series from the system labeled as past_past_feat_dynamic_real.
        # The system labels them as past_feat_dynamic_real and the additional past_ is added to the string
        # in the ForkingSequenceSplitter
        if self.use_past_feat_dynamic_real:
            # Stack features from ForkingSequenceSplitter horizontally since they were transposed
            # so shape is now (enc_len, num_past_feature_dynamic)
            chain.append(
                VstackFeatures(
                    output_field=FieldName.PAST_FEAT_DYNAMIC,
                    input_fields=[
                        "past_" + FieldName.PAST_FEAT_DYNAMIC_REAL,
                        FieldName.PAST_FEAT_DYNAMIC,
                    ],
                    h_stack=True,
                )
            )

        return Chain(chain)

    def create_training_data_loader(
        self,
        data: Dataset,
        module: MQCNNLightningModule,
        shuffle_buffer_length: Optional[int] = None,
        **kwargs,
    ) -> Iterable:
        """Creates data loader for the training dataset

        Args:
            data (Dataset): training dataset

        Returns:
            DataLoader: training data loader
        """
        train_transformation = (
            self.create_transformation()
            + self._create_instance_splitter("training")
        )

        data = Cyclic(data).stream()
        transformed_data = train_transformation.apply(data)
        return as_stacked_batches(
            transformed_data,
            batch_size=self.batch_size,
            shuffle_buffer_length=shuffle_buffer_length,
            field_names=TRAINING_INPUT_NAMES,
            output_type=torch.tensor,
            num_batches_per_epoch=self.num_batches_per_epoch,
        )

    def create_validation_data_loader(
        self,
        data: Dataset,
        module: MQCNNLightningModule,
        **kwargs,
    ) -> Iterable:
        """Creates data loader for the validation dataset

        Args:
            data (Dataset): validation dataset

        Returns:
            DataLoader: validation data loader
        """

        train_transformation = (
            self.create_transformation()
            + self._create_instance_splitter("validation")
        )

        transformed_data = train_transformation.apply(data)
        return as_stacked_batches(
            transformed_data,
            batch_size=self.val_batch_size,
            field_names=TRAINING_INPUT_NAMES,
            output_type=torch.tensor,
        )

    def create_lightning_module(self) -> MQCNNLightningModule:
        return MQCNNLightningModule(
            lr=self.lr,
            learning_rate_decay_factor=self.learning_rate_decay_factor,
            minimum_learning_rate=self.minimum_learning_rate,
            weight_decay=self.weight_decay,
            patience=self.patience,
            model_kwargs={
                "context_length": self.context_length,
                "prediction_length": self.prediction_length,
                "num_forking": self.num_forking,
                "past_feat_dynamic_real_dim": self.past_feat_dynamic_real_dim,
                "feat_dynamic_real_dim": self.feat_dynamic_real_dim,
                "cardinality_dynamic": self.cardinality_dynamic,
                "embedding_dimension_dynamic": self.embedding_dimension_dynamic,
                "feat_static_real_dim": self.feat_static_real_dim,
                "cardinality_static": self.cardinality_static,
                "embedding_dimension_static": self.embedding_dimension_static,
                "scaling": self.scaling,
                "scaling_decoder_dynamic_feature": self.scaling_decoder_dynamic_feature,
                "encoder_cnn_init_dim": self.enc_cnn_init_dim,
                "dilation_seq": self.dilation_seq,
                "kernel_size_seq": self.kernel_size_seq,
                "channels_seq": self.channels_seq,
                "joint_embedding_dimension": self.joint_embedding_dimension,
                "encoder_mlp_init_dim": self.enc_mlp_init_dim,
                "encoder_mlp_dim_seq": self.encoder_mlp_dim_seq,
                "use_residual": self.use_residual,
                "decoder_mlp_dim_seq": self.decoder_mlp_dim_seq,
                "decoder_hidden_dim": self.decoder_hidden_dim,
                "decoder_future_init_dim": self.dec_future_init_dim,
                "decoder_future_embedding_dim": self.decoder_future_embedding_dim,
                "distr_output": self.distr_output,
            },
        )

    def create_predictor(
        self,
        transformation: Transformation,
        module: MQCNNLightningModule,
    ) -> PyTorchPredictor:
        """Creates predictor for inference

        Args:
            transformation (Transformation): transformation to be applied to data input to predictor
            trained_network (MQCnnModel): trained network

        Returns:
            Predictor:
        """

        # For inference, the transformation is needed to add "future" parts of the time-series features
        prediction_splitter = self._create_instance_splitter("test")

        return PyTorchPredictor(
            input_transform=transformation + prediction_splitter,
            input_names=PREDICTION_INPUT_NAMES,
            prediction_net=module,
            batch_size=self.batch_size,
            prediction_length=self.prediction_length,
            device="auto",
            forecast_generator=self.distr_output.forecast_generator,
        )
