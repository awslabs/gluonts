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
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from gluonts.core.component import validated
from gluonts.dataset.common import load_datasets
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.loader import PseudoShuffled, TrainDataLoader
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.nursery.ContextFIDExperiment.FID import calculate_fid
from gluonts.torch.batchify import batchify
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.transform import (
    Chain,
    ExpectedNumInstanceSampler,
    InstanceSplitter,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S]",
)
logger = logging.getLogger(__name__)


class ExperimentSetUp:
    @validated()
    def __init__(
        self,
        gan_model_path: Path,
        shuffle_buffer_length: int = 5,
        train_data: bool = False,
    ) -> None:
        self.gan_model_path = gan_model_path
        self.yaml_data = self._load_yaml_data()
        self.pytorch_predictor = self._load_PytorchPredictor()
        self.pseudo_shuffle_DL = self._set_PseudoShuffledDL(
            shuffle_buffer_length, train_data
        )

    def _convert_ds_name(
        self, ds_name: str,
    ):
        ds_to_ds = {
            "m4_hourly": "/home/ec2-user/SageMaker/gluon-ts-gan/scaled_dataset/m4_hourly",
            "electricity": "/home/ec2-user/SageMaker/gluon-ts-gan/scaled_dataset/electricity_nips_scaled",
            "solar-energy": "/home/ec2-user/SageMaker/gluon-ts-gan/scaled_dataset/solar_nips_scaled",
            "exchange_rate": "/home/ec2-user/SageMaker/gluon-ts-gan/scaled_dataset/exchange_rate_nips_scaled",
            "traffic": "/home/ec2-user/SageMaker/gluon-ts-gan/scaled_dataset/traffic_nips_scaled",
        }
        return Path(ds_to_ds[ds_name])

    def _load_yaml_data(self,):
        parsed_yaml_file = yaml.load(
            open(self.gan_model_path / "data.yml"), Loader=yaml.FullLoader
        )
        return parsed_yaml_file

    def _load_PytorchPredictor(self,):
        pp = PyTorchPredictor.deserialize(
            self.gan_model_path, device=torch.device("cpu")
        )
        return pp

    def _load_dataset(self,):
        if self.yaml_data["scaling"] == "NoScale":
            ds_path = self._convert_ds_name(self.yaml_data["dataset"])
            dataset = load_datasets(
                metadata=ds_path / "metadata",
                train=ds_path / "train",
                test=ds_path / "test",
            )
            return dataset
        else:
            return get_dataset(self.yaml_data["dataset"])

    def _change_transformation(self):
        small_t = [
            k
            for k in self.pytorch_predictor.input_transform.transformations[
                :-1
            ]
        ]
        IS = InstanceSplitter(
            target_field=FieldName.TARGET,
            is_pad_field=FieldName.IS_PAD,
            start_field=FieldName.START,
            forecast_start_field=FieldName.FORECAST_START,
            instance_sampler=ExpectedNumInstanceSampler(
                num_instances=1, min_future=self.yaml_data["target_len"],
            ),
            past_length=self.yaml_data["target_len"],
            future_length=self.yaml_data["target_len"],
            time_series_fields=[
                FieldName.FEAT_TIME,
                FieldName.OBSERVED_VALUES,
            ],
        )
        small_t.append(IS)
        return Chain(small_t)

    def _load_TrainDataLoader(
        self, train_data: bool = False,
    ):
        dataset = self._load_dataset()
        TrainDL = TrainDataLoader(
            dataset=dataset.train if train_data else dataset.test,
            transform=self._change_transformation(),
            batch_size=self.pytorch_predictor.batch_size,
            stack_fn=lambda data: batchify(
                data, self.pytorch_predictor.device
            ),
        )
        return TrainDL

    def _set_PseudoShuffledDL(
        self, shuffle_buffer_length: int = 5, train_data: bool = False
    ):
        TrainDL = iter(self._load_TrainDataLoader(train_data))
        PseudoShuffleTrainDL = PseudoShuffled(TrainDL, shuffle_buffer_length)
        return PseudoShuffleTrainDL


class Experiment:
    @validated()
    def __init__(
        self,
        model_similarity_score: nn.Module,
        gan_model_path: Path,
        shuffle_buffer_length: int = 5,
        train_data: bool = False,
    ) -> None:
        self.model_similarity_score = model_similarity_score.eval()
        self.experiment_setup = ExperimentSetUp(
            gan_model_path=gan_model_path,
            shuffle_buffer_length=shuffle_buffer_length,
            train_data=train_data,
        )
        self.pytorch_predictor = self.experiment_setup.pytorch_predictor
        self.yaml_data = self.experiment_setup.yaml_data
        self.pseudo_shuffle_DL = self.experiment_setup.pseudo_shuffle_DL

    def _sample_ts(self, PseudoTrainDL: PseudoShuffled):
        batch = next(PseudoTrainDL)
        with torch.no_grad():
            synthetic_ts = self.pytorch_predictor.prediction_net(
                past_target=batch["past_target"],
                time_features=batch["future_time_feat"],
                feat_static_cat=batch["feat_static_cat"],
            ).detach()
            if self.yaml_data["which_model"] == "TIMEGAN":
                synthetic_ts = synthetic_ts.permute(0, 2, 1)
            synthetic_ts = synthetic_ts.squeeze(1)
        real_ts = batch["future_target"]
        return real_ts, synthetic_ts

    def _embed_ts(self, ts: torch.Tensor):
        with torch.no_grad():
            embed_ts = self.model_similarity_score(ts.unsqueeze(1)).detach()
        return embed_ts

    def _calculate_FID(self, PseudoTrainDL: PseudoShuffled):
        real_ts, synthetic_ts = self._sample_ts(PseudoTrainDL)
        real_embed, synthetic_embed = (
            self._embed_ts(real_ts),
            self._embed_ts(synthetic_ts),
        )
        real_fid = calculate_fid(synthetic_ts, real_ts)
        embed_fid = calculate_fid(synthetic_embed, real_embed)
        return real_fid, embed_fid

    def run_FID(
        self, nb_run: int = 10,
    ):
        logger.info("Starting experiment for the Transformer")
        fid_ts = torch.empty(nb_run)
        fid_embed = torch.empty(nb_run)
        PseudoTrainDL = iter(self.pseudo_shuffle_DL)

        for k in range(nb_run):
            logger.info(f"Running experiment number : {k}")
            fid_ts[k], fid_embed[k] = self._calculate_FID(PseudoTrainDL)

        logger.info(
            f"FID on time series: Mean = {torch.mean(fid_ts).item()}, Std = {torch.std(fid_ts).item()}"
        )
        logger.info(
            f"FID on embedding: Mean = {torch.mean(fid_embed).item()}, Std = {torch.std(fid_embed).item()}"
        )
        return fid_ts, fid_embed
