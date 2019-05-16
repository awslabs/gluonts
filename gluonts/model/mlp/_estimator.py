# Standard library imports
import itertools
from typing import List, Optional

# Third-party imports
from mxnet.gluon import HybridBlock

# First-party imports
from gluonts import transform
from gluonts.block.feature import (
    FeatureAssembler,
    FeatureEmbedder,
)  # noqa: F401
from gluonts.core.component import validated
from gluonts.distribution import DistributionOutput, StudentTOutput
from gluonts.model.estimator import GluonEstimator
from gluonts.model.predictor import Predictor, RepresentableBlockPredictor
from gluonts.trainer import Trainer
from gluonts.transform import (
    ExpectedNumInstanceSampler,
    FieldName,
    Transformation,
)

# Relative imports
from ._network import MLPLayerConfig, MLPPredictionNetwork, MLPTrainingNetwork


class MLPEstimator(GluonEstimator):
    @validated()
    def __init__(
        self,
        freq: str,
        prediction_length: int,
        trainer: Trainer = Trainer(),
        layer_configs: List[MLPLayerConfig] = [
            MLPLayerConfig(units=40, activation='linear')
        ],
        distr_output: DistributionOutput = StudentTOutput(),
        num_eval_samples: int = 100,
        embed_static: Optional[FeatureEmbedder] = None,
        embed_dynamic: Optional[FeatureEmbedder] = None,
        feat_static_cat: bool = False,
        feat_static_real: bool = False,
        feat_dynamic_cat: bool = False,
        feat_dynamic_real: bool = False,
        feat_dynamic_const: bool = True,
        feat_dynamic_age: bool = True,
    ) -> None:
        super().__init__(trainer=trainer)

        assert (
            prediction_length > 0
        ), "The value of `prediction_length` should be > 0"

        self.layer_configs = layer_configs
        self.prediction_length = prediction_length
        self.freq = freq
        self.distr_output = distr_output
        self.num_samples = num_eval_samples

        # FIXME: if synthetic categorical features are used in conjunction
        # FIXME: with an embedder,
        # FIXME: one has to adapt the corresponding FeatureEmbedder.Config
        # FIXME: accordingly
        self.embed_static = embed_static
        self.embed_dynamic = embed_dynamic

        # infer static categorical feature names
        self.feat_static_cat_names = list(
            itertools.chain(
                [FieldName.FEAT_STATIC_CAT] if feat_static_cat else []
            )
        )
        # infer static real-valued feature names
        self.feat_static_real_names = list(
            itertools.chain(
                [FieldName.FEAT_STATIC_REAL] if feat_static_real else []
            )
        )
        # infer dynamic categorical feature names
        self.feat_dynamic_cat_names = list(
            itertools.chain(
                [FieldName.FEAT_DYNAMIC_CAT] if feat_dynamic_cat else []
            )
        )
        # infer dynamic real-valued feature names
        self.feat_dynamic_real_names = list(
            itertools.chain(
                [FieldName.FEAT_CONST] if feat_dynamic_const else [],
                [FieldName.FEAT_AGE] if feat_dynamic_age else [],
                [FieldName.FEAT_DYNAMIC_REAL] if feat_dynamic_real else [],
            )
        )
        # infer all feature names
        self.feat_names = list(
            itertools.chain(
                self.feat_static_cat_names,
                self.feat_static_real_names,
                self.feat_dynamic_cat_names,
                self.feat_dynamic_real_names,
            )
        )

        if not self.feat_names:
            # FIXME: Define and throw a flavor of GluonTSHyperparametersError
            # FIXME: which does not wrap Pydantic errors
            raise RuntimeError(
                "The MLPEstimator should use at least one feature"
            )

    def create_transformation(self) -> Transformation:
        return transform.Chain(
            trans=list(
                itertools.chain(
                    # configure static categorical features
                    [
                        transform.ConcatFeatures(
                            output_field=FieldName.FEAT_STATIC_CAT,
                            input_fields=self.feat_static_cat_names,
                        )
                    ]
                    if self.feat_static_cat_names
                    else [
                        transform.SetField(
                            output_field=FieldName.FEAT_STATIC_CAT, value=[0.0]
                        ),
                        transform.AsNumpyArray(
                            field=FieldName.FEAT_STATIC_CAT, expected_ndim=1
                        ),
                    ],
                    # configure static real-valued features
                    [
                        transform.ConcatFeatures(
                            output_field=FieldName.FEAT_STATIC_REAL,
                            input_fields=self.feat_static_real_names,
                        )
                    ]
                    if self.feat_static_real_names
                    else [
                        transform.SetField(
                            output_field=FieldName.FEAT_STATIC_REAL,
                            value=[0.0],
                        ),
                        transform.AsNumpyArray(
                            field=FieldName.FEAT_STATIC_REAL, expected_ndim=1
                        ),
                    ],
                    # configure dynamic categorical features
                    [
                        transform.VstackFeatures(
                            output_field=FieldName.FEAT_DYNAMIC_CAT,
                            input_fields=self.feat_dynamic_cat_names,
                        )
                    ]
                    if self.feat_dynamic_cat_names
                    else [
                        transform.AddConstFeature(
                            output_field=FieldName.FEAT_DYNAMIC_CAT,
                            target_field=FieldName.TARGET,
                            pred_length=self.prediction_length,
                            const=0.0,
                        )
                    ],
                    # configure dynamic real-valued features
                    [
                        transform.AddConstFeature(
                            output_field=FieldName.FEAT_CONST,
                            target_field=FieldName.TARGET,
                            pred_length=self.prediction_length,
                            const=1.0,
                        )
                    ]
                    if FieldName.FEAT_CONST in self.feat_dynamic_real_names
                    else [],
                    [
                        transform.AddAgeFeature(
                            target_field=FieldName.TARGET,
                            output_field=FieldName.FEAT_AGE,
                            pred_length=self.prediction_length,
                            log_scale=False,
                        )
                    ]
                    if FieldName.FEAT_AGE in self.feat_dynamic_real_names
                    else [],
                    [
                        transform.VstackFeatures(
                            output_field=FieldName.FEAT_DYNAMIC_REAL,
                            input_fields=self.feat_dynamic_real_names,
                        )
                    ]
                    if self.feat_dynamic_real_names
                    else [
                        transform.AddConstFeature(
                            output_field=FieldName.FEAT_DYNAMIC_REAL,
                            target_field=FieldName.TARGET,
                            pred_length=self.prediction_length,
                            const=0.0,
                        )
                    ],
                    # configure instance splitter and final feature names
                    [
                        transform.CanonicalInstanceSplitter(
                            target_field=FieldName.TARGET,
                            is_pad_field=FieldName.IS_PAD,
                            start_field=FieldName.START,
                            forecast_start_field=FieldName.FORECAST_START,
                            instance_sampler=ExpectedNumInstanceSampler(
                                num_instances=4
                            ),
                            instance_length=self.prediction_length,
                            time_series_fields=[
                                FieldName.FEAT_DYNAMIC_CAT,
                                FieldName.FEAT_DYNAMIC_REAL,
                            ],
                            output_NTC=True,
                            use_prediction_features=False,
                        )
                    ],
                )
            )
        )

    def create_training_network(self) -> HybridBlock:
        # noinspection PyTypeChecker
        return MLPTrainingNetwork(
            layer_configs=self.layer_configs,
            prediction_length=self.prediction_length,
            assemble_feature=FeatureAssembler(
                T=self.prediction_length,
                use_static_cat=bool(self.feat_static_cat_names),
                use_static_real=bool(self.feat_static_real_names),
                use_dynamic_cat=bool(self.feat_dynamic_cat_names),
                use_dynamic_real=bool(self.feat_dynamic_real_names),
                embed_static=self.embed_static,
                embed_dynamic=self.embed_dynamic,
                prefix="assemble_feature_",
            ),
            distr_output=self.distr_output,
            num_samples=self.num_samples,
        )

    def create_predictor(
        self, transformation: Transformation, trained_network: HybridBlock
    ) -> Predictor:
        # noinspection PyTypeChecker
        prediction_network = MLPPredictionNetwork(
            layer_configs=self.layer_configs,
            prediction_length=self.prediction_length,
            assemble_feature=FeatureAssembler(
                T=self.prediction_length,
                # use_static_cat=bool(self.feat_static_cat_names),
                # use_static_real=bool(self.feat_static_real_names),
                # use_dynamic_cat=bool(self.feat_dynamic_cat_names),
                # use_dynamic_real=bool(self.feat_dynamic_real_names),
                embed_static=self.embed_static,
                embed_dynamic=self.embed_dynamic,
                prefix="assemble_feature_",
            ),
            distr_output=self.distr_output,
            num_samples=self.num_samples,
            params=trained_network.collect_params(),
        )

        return RepresentableBlockPredictor(
            input_transform=transformation,
            prediction_net=prediction_network,
            batch_size=self.trainer.batch_size,
            freq=self.freq,
            prediction_length=self.prediction_length,
            ctx=self.trainer.ctx,
        )
