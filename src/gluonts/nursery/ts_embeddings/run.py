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


import os
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from gluonts.nursery.ts_embeddings.embed_model import EmbedModel


class MyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        ts_len: int,
        dataset_name: str,
        num_workers: int,
        batch_size: int,
        **kwargs,
    ):
        super().__init__()
        self.ts_len = ts_len
        self.dataset_name = dataset_name
        self.num_workers = num_workers
        self.batch_size = batch_size

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--ts_len", type=int)
        parser.add_argument("--dataset_name", type=str, default=None)
        parser.add_argument("--num_workers", type=int, default=4)
        parser.add_argument("--batch_size", type=int, default=128)
        return parser

    def _get_target_tensor(self, it):
        data = []
        n_skipped = 0
        for ts in it:
            target = ts["target"]
            if len(target) < self.ts_len:
                n_skipped += 1
                continue
            data.append(target[: self.ts_len])
        if n_skipped > 0:
            print(
                f"skipped {n_skipped} series, because the target was smaller than ts_len={self.ts_len}"
            )
        print(f"num series {len(data)}")
        assert len(data) > 0
        return torch.tensor(data, dtype=torch.float32)

    def setup(self, stage=None):
        from gluonts.dataset.repository import datasets

        self.meta, self.train_ds, self.test_ds = datasets.get_dataset(
            self.dataset_name
        )
        self.X_train = self._get_target_tensor(self.train_ds)
        # self.X_test = self._get_target_tensor(test_ds)

    def train_dataloader(self):
        return DataLoader(
            # TensorDataset(torch.tensor(self.X_train)),
            self.X_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )


if __name__ == "__main__":
    parser = ArgumentParser()

    parser = MyDataModule.add_model_specific_args(parser)
    parser = EmbedModel.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    print(f"args={args}")
    data_module = MyDataModule.from_argparse_args(args)
    data_module.setup()

    model_params = vars(args)
    print(model_params)
    model = EmbedModel(**model_params)

    # from pytorch_lightning.callbacks import LearningRateLogger
    # lr_logger = LearningRateLogger(logging_interval=None)

    if "TRAINING_JOB_NAME" in os.environ:
        training_job_name = os.environ["TRAINING_JOB_NAME"]
    else:
        training_job_name = "local_test"

    # logger = TensorBoardLogger(log_path, name='embedding')
    logger = TensorBoardLogger("./logs/", name="embedding")

    trainer = Trainer.from_argparse_args(args, callbacks=[], logger=logger)

    ####  training of encoder
    trainer.fit(model, data_module)
    print("encoder training done")

    for batch in data_module.train_dataloader():
        with torch.no_grad():
            encoder = torch.jit.trace(model, batch)
        break

    encoder_model_path = "./encoder.pt"
    print(f"saving encoder to {encoder_model_path}")
    encoder.save(encoder_model_path)
    # loaded_encoder = torch.jit.load(encoder_model_path, map_location=torch.device('cuda'))
    print("loading encoder")
    loaded_encoder = torch.jit.load(encoder_model_path)

    print("running encoder")
    # ts = next(iter(data_module.train_ds))
    # target = ts['target'][:10000]
    # x = torch.tensor([target])
    with torch.no_grad():
        embed = loaded_encoder(batch).numpy()
    from sklearn.manifold import TSNE

    tsne_out = TSNE(n_components=2).fit_transform(embed)
    import matplotlib.pyplot as plt

    plt.scatter(tsne_out[:, 0], tsne_out[:, 1])
    plt.show()
