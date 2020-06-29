from torch.optim import Adam
from typing import Optional, Sequence
from box import Box

import numpy as np
import torch
from pytorch_lightning import LightningModule
from torch import nn

from gluonts.dataset.field_names import FieldName
from gluonts.dataset.loader import (
    InferenceDataLoader,
    TrainDataLoader,
    ValidationDataLoader,
)
from gluonts.time_feature import time_features_from_frequency_str
from gluonts.transform import (
    AddObservedValuesIndicator,
    AddAgeFeature,
    AddTimeFeatures,
    AsNumpyArray,
    Chain,
    CanonicalInstanceSplitter,
    ExpandDimArray,
    RemoveFields,
    SetField,
    TestSplitSampler,
    ExpectedNumInstanceSampler,
    VstackFeatures,
)
from gluonts.dataset.repository.datasets import dataset_recipes
from gluonts.dataset.common import ListDataset
from gluonts.dataset.common import load_datasets
from gluonts.dataset.repository.datasets import get_dataset as get_dataset_gts

from models_new_will_replace.gls_rbsmc import GaussianLinearSystemRBSMC
from models_new_will_replace.dynamical_system import Prediction, Latents
from experiments.base_config import TimeFeatType

from data.gluonts_nips_datasets.gluonts_nips_datasets import get_dataset
from models.gls_parameters.issm import CompositeISSM


def create_input_transform(
    is_train,
    prediction_length,
    past_length,
    use_feat_static_cat,
    use_feat_dynamic_real,
    freq,
    time_features,
    extract_tail_chunks_for_train: bool = False,
):
    SEASON_INDICATORS_FIELD = "seasonal_indicators"
    remove_field_names = [
        FieldName.FEAT_DYNAMIC_CAT,
        FieldName.FEAT_STATIC_REAL,
    ]
    if not use_feat_dynamic_real:
        remove_field_names.append(FieldName.FEAT_DYNAMIC_REAL)

    time_features = (
        time_features
        if time_features is not None
        else time_features_from_frequency_str(freq)
    )

    transform = Chain(
        [RemoveFields(field_names=remove_field_names)]
        + (
            [SetField(output_field=FieldName.FEAT_STATIC_CAT, value=[0.0])]
            if not use_feat_static_cat
            else []
        )
        + [
            AsNumpyArray(field=FieldName.FEAT_STATIC_CAT, expected_ndim=1),
            AsNumpyArray(field=FieldName.TARGET, expected_ndim=1),
            # gives target the (1, T) layout
            ExpandDimArray(field=FieldName.TARGET, axis=0),
            AddObservedValuesIndicator(
                target_field=FieldName.TARGET,
                output_field=FieldName.OBSERVED_VALUES,
            ),
            # Unnormalized seasonal features
            AddTimeFeatures(
                time_features=CompositeISSM.seasonal_features(freq),
                pred_length=prediction_length,
                start_field=FieldName.START,
                target_field=FieldName.TARGET,
                output_field=SEASON_INDICATORS_FIELD,
            ),
            AddTimeFeatures(
                start_field=FieldName.START,
                target_field=FieldName.TARGET,
                output_field=FieldName.FEAT_TIME,
                time_features=time_features,
                pred_length=prediction_length,
            ),
            AddAgeFeature(
                target_field=FieldName.TARGET,
                output_field=FieldName.FEAT_AGE,
                pred_length=prediction_length,
                log_scale=True,
            ),
            VstackFeatures(
                output_field=FieldName.FEAT_TIME,
                input_fields=[FieldName.FEAT_TIME, FieldName.FEAT_AGE]
                + (
                    [FieldName.FEAT_DYNAMIC_REAL]
                    if use_feat_dynamic_real
                    else []
                ),
            ),
            CanonicalInstanceSplitter(
                target_field=FieldName.TARGET,
                is_pad_field=FieldName.IS_PAD,
                start_field=FieldName.START,
                forecast_start_field=FieldName.FORECAST_START,
                instance_sampler=ExpectedNumInstanceSampler(num_instances=1),
                time_series_fields=[
                    FieldName.FEAT_TIME,
                    SEASON_INDICATORS_FIELD,
                    FieldName.OBSERVED_VALUES,
                ],
                allow_target_padding=True,
                instance_length=past_length,
                use_prediction_features=True,
                prediction_length=prediction_length,
            )
            if (is_train and not extract_tail_chunks_for_train)
            else CanonicalInstanceSplitter(
                target_field=FieldName.TARGET,
                is_pad_field=FieldName.IS_PAD,
                start_field=FieldName.START,
                forecast_start_field=FieldName.FORECAST_START,
                instance_sampler=TestSplitSampler(),
                time_series_fields=[
                    FieldName.FEAT_TIME,
                    SEASON_INDICATORS_FIELD,
                    FieldName.OBSERVED_VALUES,
                ],
                allow_target_padding=True,
                instance_length=past_length,
                use_prediction_features=True,
                prediction_length=prediction_length,
            ),
        ]
    )
    return transform


class GluontsUnivariateDataLoaderWrapper:
    """
    Wraps a gluonTS wrapper such that the resulting iterator yield batches
    - only of the relevant data, without all the additional stuff.
    - each as torch.Tensor,
    - with TBF format (Time, Batch, Feature).
    """

    def __init__(self, gluonts_loader):
        self._gluonts_loader = gluonts_loader

        self._all_data_keys = [
            "feat_static_cat",
            "past_target",
            "past_seasonal_indicators",
            "past_time_feat",
            "future_target",
            "future_seasonal_indicators",
            "future_time_feat",
        ]
        self._static_data_keys = [
            "feat_static_cat",
        ]

    def __iter__(self):
        for batch_gluonts in self._gluonts_loader:
            yield self._to_time_first(
                self._to_pytorch(self._extract_relevant_data(batch_gluonts,))
            )

    def _extract_relevant_data(self, gluonts_batch: dict):
        return {
            key: gluonts_batch[key]
            for key in self._all_data_keys
            if key in gluonts_batch
        }

    def _to_pytorch(self, gluonts_batch: dict):
        return {
            key: torch.tensor(gluonts_batch[key].asnumpy())
            for key, val in gluonts_batch.items()
        }

    def _to_time_first(self, torch_batch):
        return {
            key: val.transpose(1, 0)
            if key not in self._static_data_keys
            else val
            for key, val in torch_batch.items()
        }


class GluontsUnivariateDataModel(LightningModule):
    # TODO: let this take a config / hyperparam file with Hydra.
    def __init__(
        self,
        ssm: GaussianLinearSystemRBSMC,  # TODO: what about kvae?
        ctrl_transformer: nn.Module,  # TODO: make API
        tar_transformer: torch.distributions.AffineTransform,
        dataset_name: str,
        lr,
        weight_decay,
        n_epochs,
        batch_sizes,
        past_length,
        prediction_length_full,
        prediction_length_rolling,
        num_batches_per_epoch=50,
        extract_tail_chunks_for_train: bool = False,
        val_full_length=True,
    ):
        super().__init__()
        self.tar_transformer = tar_transformer
        self.ctrl_transformer = ctrl_transformer
        self.ssm = ssm
        self.dataset_name = dataset_name
        self.past_length = past_length
        self.extract_tail_chunks_for_train = extract_tail_chunks_for_train
        # TODO: naming of prediction and forecast confusing gluonts and ssm.
        self.prediction_length_rolling = prediction_length_rolling
        self.prediction_length_full = prediction_length_full
        self.val_full_length = val_full_length
        self.batch_sizes = batch_sizes
        self.num_batches_per_epoch = num_batches_per_epoch

        self.lr = lr
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs

    def forward(
        self,
        feat_static_cat: torch.Tensor,
        past_seasonal_indicators: torch.Tensor,
        past_time_feat: torch.Tensor,
        past_target: torch.Tensor,
        future_seasonal_indicators: [torch.Tensor] = None,
        future_time_feat: Optional[torch.Tensor] = None,
        future_target: Optional[torch.Tensor] = None,
        n_steps_forecast: int = 0,
        deterministic=False,
    ) -> (Sequence[Prediction], Sequence[Latents]):
        past_target = self.tar_transformer.inv(past_target)
        past_controls = self.ctrl_transformer(
            feat_static_cat=feat_static_cat,
            past_seasonal_indicators=past_seasonal_indicators,
            past_time_feat=past_time_feat,
        )
        future_controls = (
            self.ctrl_transformer(
                feat_static_cat=feat_static_cat,
                seasonal_indicators=future_seasonal_indicators,
                time_feat=future_time_feat,
            )
            if future_time_feat is not None
            else None
        )
        predictions = self.ssm.predict(
            n_steps_forecast=n_steps_forecast,
            past_targets=past_target,
            past_controls=past_controls,
            future_controls=future_controls,
            deterministic=deterministic,
        )
        # Post-process Sequence of Prediction objects
        # TODO: currently the tar_transformer transform is only used for
        #  normalisation and de-normalization of targets, i.e., we use a scale
        #  and shift transform; *In case of Gaussian likelihood* no need to
        #  correct through  change of variables (using log-abs-det).
        #  But we could use an arbitrary bijection for any likelihood function?
        for t in range(len(predictions)):
            predictions[t].emissions = self.tar_transformer(
                predictions[t].emissions
            )
        return predictions

    def prepare_data(self):
        self.dataset = get_dataset(self.dataset_name)

        input_transforms = {}
        for name in ["train", "val", "test_full", "test_rolling"]:
            if name == "train":
                prediction_length = 0
                is_train = True
                past_length_ = self.past_length
            elif name == "val":
                prediction_length = 0
                is_train = False
                past_length_ = self.past_length + (
                    self.prediction_length_full
                    if self.val_full_length
                    else self.prediction_length_rolling
                )
            elif name == "test_full":
                prediction_length = self.prediction_length_full
                is_train = False
                past_length_ = self.past_length
            elif name == "test_rolling":
                prediction_length = self.prediction_length_rolling
                is_train = False
                past_length_ = self.past_length
            else:
                raise Exception(f"unknown dataset: {name}")
            input_transforms[name] = create_input_transform(
                is_train=is_train,
                prediction_length=prediction_length,
                past_length=past_length_,
                use_feat_static_cat=True,
                use_feat_dynamic_real=True
                if self.dataset.metadata.feat_dynamic_real
                else False,
                freq=self.dataset.metadata.freq,
                time_features=None,
                extract_tail_chunks_for_train=self.extract_tail_chunks_for_train,
            )
        self.input_transforms = input_transforms

    def train_dataloader(self):
        return GluontsUnivariateDataLoaderWrapper(
            TrainDataLoader(
                dataset=self.dataset.train,
                transform=self.input_transforms["train"],
                num_batches_per_epoch=self.num_batches_per_epoch,
                batch_size=self.batch_sizes["train"],
                num_workers=0,  # TODO: had problems with > 0. Maybe they fixed
                ctx=None,
                dtype=np.float32,
            )
        )

    # def val_dataloader(self):
    #     return GluontsUnivariateDataLoaderWrapper(
    #         TrainDataLoader(
    #             dataset=self.dataset.train,
    #             transform=self.input_transforms["val"],
    #             batch_size=self.batch_sizes["val"],
    #             num_workers=0,
    #             ctx=None,
    #             dtype=np.float32,
    #         )
    #     )

    def configure_optimizers(self):
        optimizer = Adam(
            params=self.parameters(),
            lr=self.lr,
            betas=(0.9, 0.95),
            amsgrad=False,
            weight_decay=self.weight_decay,
        )
        n_iter_lr_decay_one_oom = max(int(self.n_epochs / 2), 1)
        decay_rate = (1 / 10) ** (1 / n_iter_lr_decay_one_oom)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=decay_rate,
        )
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        return {"loss": self.loss(**batch)}

    def loss(
        self,
        feat_static_cat: torch.Tensor,
        past_seasonal_indicators: torch.Tensor,
        past_time_feat: torch.Tensor,
        past_target: torch.Tensor,
    ):
        past_target = self.tar_transformer.inv(past_target)
        past_controls = self.ctrl_transformer(
            feat_static_cat=feat_static_cat,
            past_seasonal_indicators=past_seasonal_indicators,
            past_time_feat=past_time_feat,
        )
        loss = self.ssm.loss(
            past_targets=past_target, past_controls=past_controls,
        )
        return loss
