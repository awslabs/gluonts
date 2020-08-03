import os
from torch.optim import Adam
from typing import Optional, Sequence, Iterator, Dict, Union, List, Tuple

import numpy as np
import torch
from torch import Tensor
from pytorch_lightning import LightningModule
import pytorch_lightning as pl
from torch.optim.optimizer import Optimizer

from gluonts.evaluation._base import Evaluator, _worker_init, _worker_fun
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.loader import (
    InferenceDataLoader,
    TrainDataLoader,
    ValidationDataLoader,
)
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.model.predictor import RepresentablePredictor
from gluonts.dataset.common import Dataset
from gluonts.model.forecast import Forecast, SampleForecast
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
from inference.smc.resampling import EffectiveSampleSizeResampleCriterion

from models_new_will_replace.base_rbpf_gls import BaseRBSMCGaussianLinearSystem
from models_new_will_replace.base_amortized_gls import BaseAmortizedGaussianLinearSystem
from models_new_will_replace.base_gls import Prediction, Latents

from data.gluonts_nips_datasets.gluonts_nips_datasets import get_dataset
from models_new_will_replace.gls_parameters.issm import CompositeISSM
from utils.utils import shorten_iter
from visualization.plot_forecasts import make_val_plots_univariate
from experiments_new_will_replace.model_component_zoo.input_transforms import (
    InputTransformer,
)


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

    def __init__(
            self,
            gluonts_loader,
            is_for_predictor: int = False,
            float_dtype: Optional[torch.dtype] = None,
    ):
        self._gluonts_loader = gluonts_loader
        self._is_for_predictor = is_for_predictor
        self.float_dtype = float_dtype
        self.int_dtype = torch.int64

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
        self._int_data_keys = [
            "past_seasonal_indicators",
            "future_seasonal_indicators",
        ]

    def __len__(self):
        # neccessary for validation if we do not want to eval all data.
        # this may be used wrong for infinite train loader....
        if hasattr(self._gluonts_loader, "__len__"):
            return len(self._gluonts_loader)
        return (self._gluonts_loader.parallel_data_loader.dataset_len
                // self._gluonts_loader.batch_size) + 1

    def __iter__(self):
        if self._is_for_predictor:
            for batch_gluonts in self._gluonts_loader:
                yield (
                    self._to_time_first(
                        self._to_pytorch(
                            self._extract_relevant_data(
                                batch_gluonts,
                            )
                        )
                    ),
                    {
                        k: v for k, v in batch_gluonts.items()
                        if k in [FieldName.ITEM_ID, "forecast_start"]
                    },
                )
        else:
            for batch_gluonts in self._gluonts_loader:
                yield self._to_time_first(
                    self._to_pytorch(
                        self._extract_relevant_data(
                            batch_gluonts,
                        )
                    )
                )

    def _extract_relevant_data(self, gluonts_batch: dict):
        return {
            key: gluonts_batch[key]
            for key in self._all_data_keys
            if key in gluonts_batch
        }

    def _to_pytorch(self, gluonts_batch: dict):
        return {
            key: torch.tensor(
                gluonts_batch[key].asnumpy(),
                dtype=self.int_dtype
                if key in self._int_data_keys else self.float_dtype
            ) for key, val in gluonts_batch.items()
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
        ssm: BaseAmortizedGaussianLinearSystem,
        ctrl_transformer: InputTransformer,
        tar_transformer: torch.distributions.AffineTransform,
        dataset_name: str,
        lr,
        weight_decay,
        n_epochs,
        batch_sizes,
        past_length,
        n_particle_train,
        n_particle_eval,
        prediction_length_full,
        prediction_length_rolling,
        n_epochs_no_resampling=0,
        n_epochs_freeze_gls_params=0,
        num_batches_per_epoch=50,
        extract_tail_chunks_for_train: bool = False,
        val_full_length=True,
        deterministic_forecast: bool = False,
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
        self.n_epochs_no_resampling = n_epochs_no_resampling
        self.n_epochs_freeze_gls_params = n_epochs_freeze_gls_params
        self.num_batches_per_epoch = num_batches_per_epoch

        self.lr = lr
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.deterministic_forecast = deterministic_forecast

        assert n_particle_train == self.ssm.n_particle
        self._n_particle_train = n_particle_train
        self._n_particle_eval = n_particle_eval

        self.forecast_evaluator = Evaluator(
            quantiles=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
            num_workers=0,  # has been buggy with > 0. Maybe check again.
        )

    def forward(
        self,
        past_target: torch.Tensor,
        feat_static_cat: Optional[torch.Tensor] = None,
        past_seasonal_indicators: Optional[torch.Tensor] = None,
        past_time_feat: Optional[torch.Tensor] = None,
        future_seasonal_indicators: Optional[torch.Tensor] = None,
        future_time_feat: Optional[torch.Tensor] = None,
        n_steps_forecast: int = 0,
        deterministic=False,
    ) -> (Sequence[Prediction], Sequence[Latents]):
        past_target = self.tar_transformer.inv(past_target)
        past_controls = self.ctrl_transformer(
            feat_static_cat=feat_static_cat,
            seasonal_indicators=past_seasonal_indicators,
            time_feat=past_time_feat,
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
        predictions_inferred, predictions_forecast = self.ssm.predict(
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
        for t in range(len(predictions_inferred)):
            predictions_inferred[t].emissions = self.tar_transformer(
                predictions_inferred[t].emissions,
            )
        for t in range(len(predictions_forecast)):
            predictions_forecast[t].emissions = self.tar_transformer(
                predictions_forecast[t].emissions,
            )
        return predictions_inferred, predictions_forecast

    def prepare_data(self):
        """
        prepare data is called only once - also if doing multi-GPU.
        A few things such as input_transform and predictors depend on data.
        So we do those here as well.
        """
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

        self.predictors = {
            "full": SGLSPredictor(
                model=self,
                input_transform=self.input_transforms['test_full'],
                batch_size=self.batch_sizes['test_full'],
                prediction_length=self.prediction_length_full,
                freq=self.dataset.metadata.freq,
            ),
            "rolling": SGLSPredictor(
                model=self,
                input_transform=self.input_transforms['test_rolling'],
                batch_size=self.batch_sizes['test_rolling'],
                prediction_length=self.prediction_length_rolling,
                freq=self.dataset.metadata.freq,
            ),
        }

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
            ),
            float_dtype=self.dtype,
        )

    def val_dataloader(self):
        return GluontsUnivariateDataLoaderWrapper(
            ValidationDataLoader(
                dataset=self.dataset.train,
                transform=self.input_transforms["val"],
                batch_size=self.batch_sizes["val"],
                num_workers=0,
                ctx=None,
            ),
            float_dtype=self.dtype,
        )

    def test_dataloader(self):
        test_loader_full = GluontsUnivariateDataLoaderWrapper(
            InferenceDataLoader(
                dataset=self.dataset.test,
                transform=self.input_transforms["test_full"],
                batch_size=self.batch_sizes["test_full"],
                num_workers=0,
                ctx=None,
                dtype=np.float32,
            ),
            float_dtype=self.dtype,
        )
        # test_loader_rolling = GluontsUnivariateDataLoaderWrapper(
        #     InferenceDataLoader(
        #         dataset=self.dataset.test,
        #         transform=self.input_transforms["test_rolling"],
        #         batch_size=self.batch_sizes["test_rolling"],
        #         num_workers=0,
        #         ctx=None,
        #         dtype=np.float32,
        #     )
        # )
        return test_loader_full

    def optimizer_step(
            self,
            epoch: int,
            batch_idx: int,
            optimizer: Optimizer,
            optimizer_idx: int,
            *args,
            **kwargs,
    ) -> None:
        # warmup only certain parameters (all except GLS) if configured.
        is_warmup = (epoch < self.n_epochs_freeze_gls_params)
        if is_warmup:
            lr_gls = optimizer.param_groups[0]['lr']
            optimizer.param_groups[0]['lr'] = 0
            optimizer_output = super().optimizer_step(
                epoch, batch_idx, optimizer, optimizer_idx, *args, **kwargs,
            )
            optimizer.param_groups[0]['lr'] = lr_gls
        else:
            optimizer_output = super().optimizer_step(
                epoch, batch_idx, optimizer, optimizer_idx, *args, **kwargs,
            )
        return optimizer_output

    def configure_optimizers(self):
        param_names_gls = [
            name
            for name in dict(self.named_parameters()).keys()
            if
            ("gls_base_parameters" in name) and (not "link_transformers" in name)
        ]
        params_gls = tuple(
            param
            for name, param in self.named_parameters()
            if name in param_names_gls
        )
        params_except_gls = tuple(
            param
            for name, param in self.named_parameters()
            if name not in param_names_gls
        )
        assert len(params_except_gls) < len(tuple(self.parameters()))

        optimizer = Adam(
            params=[
                {"params": params_gls, "lr": self.lr},
                {"params": params_except_gls, "lr": self.lr},
            ],
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
        loss = self.loss(**batch)
        result = pl.TrainResult(loss)
        result.log('train_loss', loss)
        return result

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            loss = self.loss(**batch)
        result = pl.EvalResult(checkpoint_on=loss)
        result.log('val_loss', loss)

        if isinstance(self.ssm, BaseRBSMCGaussianLinearSystem):
            if batch_idx == 0:
                # We don't get future_target from gluonTS
                # (probably there is an easy way though).
                # Instead, we split past_target into two parts.
                past_keys = [key for key in batch.keys() if "past" in key]
                for k_past in past_keys:
                    k_future = k_past.replace("past", "future")
                    tmp = batch[k_past]
                    batch[k_past] = tmp[:self.past_length, :]
                    batch[k_future] = tmp[self.past_length:, :]

                make_val_plots_univariate(
                    model=self,
                    data=batch,
                    idx_particle=None,
                    n_steps_forecast=self.prediction_length_full,
                    idxs_ts=[0, 1, 2],
                    show=False,
                    savepath=os.path.join(
                        self.trainer.default_root_dir, "plots",
                        f"forecast_ep{self.current_epoch}",
                    ),
                )
        return result

    def test_step(self, batch, batch_idx):
        # TODO: forecast metrics are computed currently in
        #  end_test / validation_epoch_end, because of streaming setting of
        #  gluonTS evaluation functions. I don't know of a proper
        #  way with reusing existing functions...
        return
        # with torch.no_grad():
        #     loss = self.loss(**batch)
        # result = pl.EvalResult()
        # result.log('test_loss', loss)
        # return result

    def on_train_start(self) -> None:
        os.makedirs(
            os.path.join(self.trainer.default_root_dir, "plot"),
            exist_ok=True,
        )

    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()
        self.ssm.n_particle = self._n_particle_train

        # Set to no re-sampling if configured.
        # Note that this will omit the very first iteration
        # since the computation happens before this function,
        # but it should not be a problem. Want to get rid of this anyways.
        if isinstance(self.ssm, BaseRBSMCGaussianLinearSystem):
            # <= and >= because we want to set self._resampling_criterion_fn.
            set_no_resampling = \
                (self.current_epoch <= self.n_epochs_no_resampling)
            set_resampling = \
                (self.current_epoch >= self.n_epochs_no_resampling)

            if set_no_resampling:
                self._resampling_criterion_fn = self.ssm.resampling_criterion_fn
                self.ssm.resampling_criterion_fn = \
                    EffectiveSampleSizeResampleCriterion(
                        min_ess_ratio=0.0,
                    )
            if set_resampling:
                self.ssm.resampling_criterion_fn = self._resampling_criterion_fn

    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()
        self.ssm.n_particle = self._n_particle_eval

    def on_test_epoch_start(self) -> None:
        super().on_test_epoch_start()
        self.ssm.n_particle = self._n_particle_eval

    def validation_epoch_end(
        self, outputs: Union[List[Dict[str, Tensor]], List[List[Dict[str, Tensor]]]]
    ) -> Dict[str, Dict[str, Tensor]]:
        # GluonTS has streaming setting and predictors take dataset objects.
        # This does not fit with validation_step working on batches.
        # Therefore, we do the forecast metrics evaluation here with a wrapper.
        dataset = self.dataset.train  # there is no val dataset -> use train

        agg_metrics = {}
        for which, predictor in self.predictors.items():
            if which == "rolling":
                continue  # its a bit expensive to validate rolling
            forecast_it, ts_it = make_evaluation_predictions(
                dataset,
                predictor=predictor,
                num_samples=self.ssm.n_particle,
            )
            assert len(self.trainer.num_val_batches) == 1
            num_series = self.trainer.num_val_batches[0] * self.batch_sizes["val"]
            if num_series >= len(dataset):
                num_series = len(dataset)
            else:
                forecast_it = shorten_iter(forecast_it, num_series)
                ts_it = shorten_iter(ts_it, num_series)

            _agg_metrics, _ = self.forecast_evaluator(
                ts_it, forecast_it, num_series=num_series,
            )
            for key, val in _agg_metrics.items():
                agg_metrics[f"{key}_{which}"] = val
        # TODO: do something with "outputs"
        result = pl.EvalResult(
            checkpoint_on=torch.tensor(agg_metrics["mean_wQuantileLoss_full"]),
        )
        for k, v in agg_metrics.items():
            prog_bar = True if k == "mean_wQuantileLoss_full" else False
            result.log(k, v, prog_bar=prog_bar)

        return result

    def test_end(
            self,
            outputs: Union[List[Dict[str, Tensor]], List[List[Dict[str, Tensor]]]]
    ) -> Dict[str, Dict[str, Tensor]]:
        dataset = self.dataset.test

        agg_metrics = {}
        for which, predictor in self.predictors.items():
            forecast_it, ts_it = make_evaluation_predictions(
                dataset,
                predictor=predictor,
                num_samples=self.ssm.n_particle,
            )

            _agg_metrics, _ = self.forecast_evaluator(
                ts_it, forecast_it, num_series=len(dataset),
            )
            for key, val in _agg_metrics.items():
                agg_metrics[f"{key}_{which}"] = val

        result = pl.EvalResult()
        for k, v in agg_metrics.items():
            result.log(k, v)
        return result

    def loss(
        self,
        feat_static_cat: torch.Tensor,
        past_seasonal_indicators: torch.Tensor,
        past_time_feat: torch.Tensor,
        past_target: torch.Tensor,
        **kwargs,
    ):
        T, B = past_target.shape[:2]

        past_target = self.tar_transformer.inv(past_target)
        past_controls = self.ctrl_transformer(
            feat_static_cat=feat_static_cat,
            seasonal_indicators=past_seasonal_indicators,
            time_feat=past_time_feat,
        )
        loss_samplewise = self.ssm.loss(
            past_targets=past_target,
            past_controls=past_controls,
            **{k: v for k, v in kwargs.items() if not "future" in k},
        )
        loss = loss_samplewise.sum(dim=0) / (T * B)
        return loss


class SGLSPredictor(RepresentablePredictor):
    """ wrapper to to allow make_evaluation_predictions to evaluate this model. """

    def __init__(
        self,
        model: GluontsUnivariateDataModel,
        input_transform,
        batch_size: int,
        prediction_length: int,
        freq: str,
        lead_time: int = 0,
    ):
        super().__init__(
            prediction_length=prediction_length, freq=freq, lead_time=lead_time,
        )
        self.model = model
        self.input_transform = input_transform
        self.batch_size = batch_size

    def predict(
            self, dataset: Dataset, **kwargs,
    ) -> Iterator[Dict[Forecast, torch.Tensor]]:
        if 'num_samples' in kwargs:
            assert kwargs.pop('num_samples') == self.model.ssm.n_particle

        inference_loader = GluontsUnivariateDataLoaderWrapper(
            InferenceDataLoader(
                dataset=dataset,
                transform=self.input_transform,
                batch_size=self.batch_size,
                num_workers=0,
                num_prefetch=0,
                ctx=None,
                dtype=np.float32,
            ),
            is_for_predictor=True,
            float_dtype=self.model.dtype,
        )
        for batch, batch_metainfo in inference_loader:  # put manually on GPU.
            batch = {k: v.to(self.model.device) for k, v in batch.items()}
            assert len(batch['future_time_feat']) == self.prediction_length

            predictions_inferred, predictions_forecast = self.model(
                **batch,
                n_steps_forecast=self.prediction_length,
                deterministic=self.model.deterministic_forecast,
            )
            forecast_gts = torch.stack(
                [fcst.emissions for fcst in predictions_forecast], dim=0,
            )
            forecast_gts = forecast_gts.transpose(0, 2)  # TPBF -> BPTF
            forecast_gts = forecast_gts.detach().cpu().numpy()
            # squeezing is bad, but gluonts backtest requires it.
            forecast_gts = forecast_gts.squeeze(axis=-1)

            for idx_sample_in_batch, _fcst_gts in enumerate(forecast_gts):
                yield SampleForecast(
                    samples=_fcst_gts,
                    start_date=batch_metainfo["forecast_start"][idx_sample_in_batch],
                    freq=self.freq,
                    item_id=batch_metainfo[FieldName.ITEM_ID][idx_sample_in_batch]
                    if FieldName.ITEM_ID in batch_metainfo else None,
                )

            assert idx_sample_in_batch + 1 == len(batch_metainfo["forecast_start"])