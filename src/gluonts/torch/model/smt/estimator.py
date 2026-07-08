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

from typing import List, Optional, Iterable, Dict, Any

import torch
import lightning.pytorch as pl

from gluonts.core.component import validated
from gluonts.env import env
from gluonts.dataset.common import Dataset
from gluonts.itertools import Cached
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.loader import as_stacked_batches
from gluonts.itertools import Cyclic
from gluonts.dataset.stat import calculate_dataset_statistics
from gluonts.time_feature import (
    TimeFeature,
    time_features_from_frequency_str,
)
from gluonts.transform import (
    Transformation,
    Chain,
    RemoveFields,
    SetField,
    AsNumpyArray,
    AddObservedValuesIndicator,
    AddTimeFeatures,
    AddAgeFeature,
    VstackFeatures,
    InstanceSplitter,
    ValidationSplitSampler,
    TestSplitSampler,
    ExpectedNumInstanceSampler,
    MissingValueImputation,
    DummyValueImputation,
)
from gluonts.torch.model.estimator import (
    PyTorchLightningEstimator,
    TrainOutput,
)
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.torch.distributions import DistributionOutput, StudentTOutput
from gluonts.transform.sampler import InstanceSampler

from .lightning_module import SMTLightningModule

PREDICTION_INPUT_NAMES = [
    "feat_static_cat",
    "feat_static_real",
    "past_time_feat",
    "past_target",
    "past_observed_values",
    "future_time_feat",
]

TRAINING_INPUT_NAMES = PREDICTION_INPUT_NAMES + [
    "future_target",
    "future_observed_values",
]


class SMTEstimator(PyTorchLightningEstimator):
    """Estimator for a forecasting model trained with Supervised Memory
    Training (SMT) [Kumar & Isola, 2026].

    The model has the same probabilistic, covariate-aware setup as
    ``DeepAREstimator`` (lags, time/age features, static embeddings, mean
    scaling and a ``DistributionOutput`` head), but the recurrence is realized
    with a Transformer encoder/masked-decoder instead of an LSTM, and is
    trained with the SMT objective (predictive-state NLL + one-step memory
    dynamics + uniformity) rather than backpropagation through time.

    Parameters
    ----------
    freq
        Frequency of the data to train on and predict.
    prediction_length
        Length of the prediction horizon.
    context_length
        Number of past steps to encode before forecasting
        (default: ``prediction_length``).
    d_model
        Transformer model dimension.
    nhead
        Number of attention heads.
    num_encoder_layers
        Number of layers in the (bidirectional) teacher encoder.
    num_decoder_layers
        Number of layers in the (causal) decoder / readout head.
    num_rnn_layers
        Number of layers in the recurrent memory cell.
    mem_tokens
        Number of memory tokens the memory state is factorized into.
    dim_feedforward
        Hidden size of the Transformer feed-forward blocks
        (default: ``4 * d_model``).
    dropout_rate
        Dropout regularization parameter.
    attn_type
        Token mixer for the bidirectional stacks (teacher encoder and recurrent
        memory cell): ``"softmax"`` (default, standard attention), or one of the
        functional-attention variants ``"funcattn"`` (Xu et al., 2026),
        ``"intention"`` or ``"linear"``. The causal decoder always uses softmax
        attention.
    num_slices
        Number of adaptive basis slices for ``attn_type="funcattn"`` (ignored
        otherwise).
    ridge
        Ridge regularization of the least-squares solve for
        ``attn_type="intention"`` (ignored otherwise).
    coef_dyn
        Weight of the one-step memory dynamics loss.
    coef_unif
        Weight of the memory uniformity regularizer.
    dmt_finetune_epochs
        If ``> 0``, run a DAgger Memory Training (DMT) finetuning phase for
        this many epochs after the SMT phase: the teacher (encoder/decoder) is
        frozen and only the recurrent cell is trained, on-policy, to track the
        teacher's memory trajectory under its own rollout (Kumar & Isola, 2026,
        Sec. 2.3). This is the full ``SMT -> DMT`` method.
    dmt_rollout_steps
        Number of steps the recurrent cell is unrolled during DMT
        (default: ``prediction_length``, i.e. the deployment horizon). Capped
        at ``prediction_length``.
    dmt_lr
        Learning rate for the DMT finetuning phase (kept small).
    lr
        Learning rate.
    weight_decay
        Weight decay regularization parameter.
    patience
        Patience parameter for the learning rate scheduler.
    distr_output
        Distribution to fit and sample from (default: ``StudentTOutput()``).
    scaling
        Whether to apply mean scaling to the target.
    num_parallel_samples
        Number of sample paths to draw per series at prediction time.
    batch_size
        Batch size used during training.
    num_batches_per_epoch
        Number of batches per training epoch.
    trainer_kwargs
        Additional arguments for the PyTorch Lightning ``Trainer``.
    """

    @validated()
    def __init__(
        self,
        freq: str,
        prediction_length: int,
        context_length: Optional[int] = None,
        d_model: int = 32,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        num_rnn_layers: int = 2,
        mem_tokens: int = 4,
        dim_feedforward: Optional[int] = None,
        dropout_rate: float = 0.1,
        attn_type: str = "softmax",
        num_slices: int = 32,
        ridge: float = 1e-3,
        coef_dyn: float = 0.1,
        coef_unif: float = 0.001,
        dmt_finetune_epochs: int = 0,
        dmt_rollout_steps: Optional[int] = None,
        dmt_lr: float = 1e-4,
        lr: float = 1e-3,
        weight_decay: float = 1e-8,
        patience: int = 10,
        num_feat_dynamic_real: int = 0,
        num_feat_static_cat: int = 0,
        num_feat_static_real: int = 0,
        cardinality: Optional[List[int]] = None,
        embedding_dimension: Optional[List[int]] = None,
        distr_output: DistributionOutput = StudentTOutput(),
        scaling: bool = True,
        default_scale: Optional[float] = None,
        lags_seq: Optional[List[int]] = None,
        time_features: Optional[List[TimeFeature]] = None,
        num_parallel_samples: int = 100,
        batch_size: int = 32,
        num_batches_per_epoch: int = 50,
        imputation_method: Optional[MissingValueImputation] = None,
        trainer_kwargs: Optional[Dict[str, Any]] = None,
        train_sampler: Optional[InstanceSampler] = None,
        validation_sampler: Optional[InstanceSampler] = None,
        nonnegative_pred_samples: bool = False,
    ) -> None:
        default_trainer_kwargs = {
            "max_epochs": 100,
            "gradient_clip_val": 10.0,
        }
        if trainer_kwargs is not None:
            default_trainer_kwargs.update(trainer_kwargs)
        super().__init__(trainer_kwargs=default_trainer_kwargs)

        self.freq = freq
        self.context_length = (
            context_length if context_length is not None else prediction_length
        )
        self.prediction_length = prediction_length
        self.patience = patience
        self.distr_output = distr_output
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.num_rnn_layers = num_rnn_layers
        self.mem_tokens = mem_tokens
        self.dim_feedforward = dim_feedforward
        self.attn_type = attn_type
        self.num_slices = num_slices
        self.ridge = ridge
        self.coef_dyn = coef_dyn
        self.coef_unif = coef_unif
        self.dmt_finetune_epochs = dmt_finetune_epochs
        self.dmt_rollout_steps = dmt_rollout_steps
        self.dmt_lr = dmt_lr
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout_rate = dropout_rate
        self.num_feat_dynamic_real = num_feat_dynamic_real
        self.num_feat_static_cat = num_feat_static_cat
        self.num_feat_static_real = num_feat_static_real
        self.cardinality = (
            cardinality if cardinality and num_feat_static_cat > 0 else [1]
        )
        self.embedding_dimension = embedding_dimension
        self.scaling = scaling
        self.default_scale = default_scale
        self.lags_seq = lags_seq
        self.time_features = (
            time_features
            if time_features is not None
            else time_features_from_frequency_str(self.freq)
        )

        self.num_parallel_samples = num_parallel_samples
        self.batch_size = batch_size
        self.num_batches_per_epoch = num_batches_per_epoch

        self.imputation_method = (
            imputation_method
            if imputation_method is not None
            else DummyValueImputation(self.distr_output.value_in_support)
        )

        self.train_sampler = train_sampler or ExpectedNumInstanceSampler(
            num_instances=1.0, min_future=prediction_length
        )
        self.validation_sampler = validation_sampler or ValidationSplitSampler(
            min_future=prediction_length
        )
        self.nonnegative_pred_samples = nonnegative_pred_samples

    @classmethod
    def derive_auto_fields(cls, train_iter):
        stats = calculate_dataset_statistics(train_iter)

        return {
            "num_feat_dynamic_real": stats.num_feat_dynamic_real,
            "num_feat_static_cat": len(stats.feat_static_cat),
            "cardinality": [len(cats) for cats in stats.feat_static_cat],
        }

    def create_transformation(self) -> Transformation:
        remove_field_names = []
        if self.num_feat_static_real == 0:
            remove_field_names.append(FieldName.FEAT_STATIC_REAL)
        if self.num_feat_dynamic_real == 0:
            remove_field_names.append(FieldName.FEAT_DYNAMIC_REAL)

        return Chain(
            [RemoveFields(field_names=remove_field_names)]
            + (
                [SetField(output_field=FieldName.FEAT_STATIC_CAT, value=[0])]
                if not self.num_feat_static_cat > 0
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
                    dtype=int,
                ),
                AsNumpyArray(
                    field=FieldName.FEAT_STATIC_REAL,
                    expected_ndim=1,
                ),
                AsNumpyArray(
                    field=FieldName.TARGET,
                    expected_ndim=1 + len(self.distr_output.event_shape),
                ),
                AddObservedValuesIndicator(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.OBSERVED_VALUES,
                    imputation_method=self.imputation_method,
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
                AsNumpyArray(FieldName.FEAT_TIME, expected_ndim=2),
            ]
        )

    def _create_instance_splitter(self, module: SMTLightningModule, mode: str):
        assert mode in ["training", "validation", "test"]

        instance_sampler = {
            "training": self.train_sampler,
            "validation": self.validation_sampler,
            "test": TestSplitSampler(),
        }[mode]

        return InstanceSplitter(
            target_field=FieldName.TARGET,
            is_pad_field=FieldName.IS_PAD,
            start_field=FieldName.START,
            forecast_start_field=FieldName.FORECAST_START,
            instance_sampler=instance_sampler,
            past_length=module.model._past_length,
            future_length=self.prediction_length,
            time_series_fields=[
                FieldName.FEAT_TIME,
                FieldName.OBSERVED_VALUES,
            ],
            dummy_value=self.distr_output.value_in_support,
        )

    def create_training_data_loader(
        self,
        data: Dataset,
        module: SMTLightningModule,
        shuffle_buffer_length: Optional[int] = None,
        **kwargs,
    ) -> Iterable:
        data = Cyclic(data).stream()
        instances = self._create_instance_splitter(module, "training").apply(
            data, is_train=True
        )
        return as_stacked_batches(
            instances,
            batch_size=self.batch_size,
            shuffle_buffer_length=shuffle_buffer_length,
            field_names=TRAINING_INPUT_NAMES,
            output_type=torch.tensor,
            num_batches_per_epoch=self.num_batches_per_epoch,
        )

    def create_validation_data_loader(
        self,
        data: Dataset,
        module: SMTLightningModule,
        **kwargs,
    ) -> Iterable:
        instances = self._create_instance_splitter(module, "validation").apply(
            data, is_train=True
        )
        return as_stacked_batches(
            instances,
            batch_size=self.batch_size,
            field_names=TRAINING_INPUT_NAMES,
            output_type=torch.tensor,
        )

    def create_lightning_module(self) -> SMTLightningModule:
        return SMTLightningModule(
            lr=self.lr,
            weight_decay=self.weight_decay,
            patience=self.patience,
            dmt_lr=self.dmt_lr,
            model_kwargs={
                "freq": self.freq,
                "context_length": self.context_length,
                "prediction_length": self.prediction_length,
                "num_feat_dynamic_real": (
                    1 + self.num_feat_dynamic_real + len(self.time_features)
                ),
                "num_feat_static_real": max(1, self.num_feat_static_real),
                "num_feat_static_cat": max(1, self.num_feat_static_cat),
                "cardinality": self.cardinality,
                "embedding_dimension": self.embedding_dimension,
                "d_model": self.d_model,
                "nhead": self.nhead,
                "num_encoder_layers": self.num_encoder_layers,
                "num_decoder_layers": self.num_decoder_layers,
                "num_rnn_layers": self.num_rnn_layers,
                "mem_tokens": self.mem_tokens,
                "dim_feedforward": self.dim_feedforward,
                "dropout_rate": self.dropout_rate,
                "attn_type": self.attn_type,
                "num_slices": self.num_slices,
                "ridge": self.ridge,
                "coef_dyn": self.coef_dyn,
                "coef_unif": self.coef_unif,
                "dmt_rollout_steps": self.dmt_rollout_steps,
                "distr_output": self.distr_output,
                "lags_seq": self.lags_seq,
                "scaling": self.scaling,
                "default_scale": self.default_scale,
                "num_parallel_samples": self.num_parallel_samples,
                "nonnegative_pred_samples": self.nonnegative_pred_samples,
            },
        )

    def create_predictor(
        self,
        transformation: Transformation,
        module: SMTLightningModule,
    ) -> PyTorchPredictor:
        prediction_splitter = self._create_instance_splitter(module, "test")

        return PyTorchPredictor(
            input_transform=transformation + prediction_splitter,
            input_names=PREDICTION_INPUT_NAMES,
            prediction_net=module,
            batch_size=self.batch_size,
            prediction_length=self.prediction_length,
            device="auto",
        )

    def train_model(
        self,
        training_data: Dataset,
        validation_data: Optional[Dataset] = None,
        from_predictor=None,
        shuffle_buffer_length: Optional[int] = None,
        cache_data: bool = False,
        ckpt_path: Optional[str] = None,
        **kwargs,
    ) -> TrainOutput:
        # phase 1: standard SMT training
        out = super().train_model(
            training_data,
            validation_data,
            from_predictor=from_predictor,
            shuffle_buffer_length=shuffle_buffer_length,
            cache_data=cache_data,
            ckpt_path=ckpt_path,
        )
        if self.dmt_finetune_epochs <= 0:
            return out

        # phase 2: DMT finetuning -- freeze the teacher, train only the
        # recurrent cell on-policy to correct its rollout drift.
        transformation = out.transformation
        net = out.trained_net
        net.enable_dmt(self.dmt_lr)

        with env._let(max_idle_transforms=max(len(training_data), 100)):
            transformed = transformation.apply(training_data, is_train=True)
            if cache_data:
                transformed = Cached(transformed)
            training_data_loader = self.create_training_data_loader(
                transformed, net, shuffle_buffer_length=shuffle_buffer_length
            )

        validation_data_loader = None
        if validation_data is not None:
            with env._let(max_idle_transforms=max(len(validation_data), 100)):
                transformed_val = transformation.apply(
                    validation_data, is_train=True
                )
                if cache_data:
                    transformed_val = Cached(transformed_val)
                validation_data_loader = self.create_validation_data_loader(
                    transformed_val, net
                )

        monitor = "train_loss" if validation_data is None else "val_loss"
        checkpoint = pl.callbacks.ModelCheckpoint(
            monitor=monitor, mode="min", verbose=True
        )
        trainer_kwargs = {
            **self.trainer_kwargs,
            "max_epochs": self.dmt_finetune_epochs,
        }
        custom_callbacks = trainer_kwargs.pop("callbacks", [])
        trainer = pl.Trainer(
            **{
                "accelerator": "auto",
                "callbacks": [checkpoint] + custom_callbacks,
                **trainer_kwargs,
            }
        )
        trainer.fit(
            model=net,
            train_dataloaders=training_data_loader,
            val_dataloaders=validation_data_loader,
        )

        if checkpoint.best_model_path != "":
            best_model = net.__class__.load_from_checkpoint(
                checkpoint.best_model_path
            )
        else:
            best_model = net

        return TrainOutput(
            transformation=transformation,
            trained_net=best_model,
            trainer=trainer,
            predictor=self.create_predictor(transformation, best_model),
        )
