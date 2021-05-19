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


from typing import Optional, List
import re
import ast
from argparse import ArgumentParser, ArgumentError, ArgumentTypeError

import numpy as np
import torch
import torch.nn.functional as F

import pytorch_lightning as pl

from . import causal_cnn
from . import contrastive_loss
from .lars_wrapper import LARSWrapper

import logging

logging.getLogger("lightning").setLevel(logging.INFO)


def try_add_arg(parser, *args, **kwargs):
    try:
        parser.add_argument(*args, **kwargs)
    except ArgumentError:
        logger = logging.getLogger(__name__)
        logger.warning(f"error adding arguments args={args}, kwargs={kwargs}")


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ArgumentTypeError("Boolean value expected.")


class PreProcessor:
    """
    A PreProcessor can be used before the embedding model is used.
    The `num_out_dim` indicates the number of outputs (e.g. 1 if the pre-processor returns scalar time series).
    This is later used as input channels in the convolution layer.
    The pre-processor should only use torch operations, since it will be stored as part of the final embedding model.

    Note: the preprocessor operates on a batch, not the entire data set.
    """

    num_out_dim: int

    def __call__(self, ts_batch):
        """
        Pre-processes a batch of time series with equal length.
        The input shape should be
        (N, T) or (N, C, T)

        The output shape should be

        (N, num_out_dim, L)
        """
        raise NotImplementedError()


class ScalePreProcessor:
    """
    A pre-processor that takes a (possibly multi-variate) time series and normalizes each dimension by its mean
    """

    def __init__(self, out_dim, min_scale=1e-12):
        self.min_scale = min_scale
        self.num_out_dim = out_dim

    def __call__(self, ts_batch):
        s = ts_batch.shape
        assert (
            len(s) == 2 or len(s) == 3
        ), "Expecting a batch of uni or multivariate time series"

        T = ts_batch.size(-1)
        scale = torch.nansum(torch.abs(ts_batch), dim=-1) / T
        scale = torch.clip(scale, self.min_scale, np.inf)

        return res[:, :, :]


class AggregationPreProcessor:
    """
    A pre-processor that aggregates the time series using agg_window length.
    agg_fns can be a quantile or mean

    "q0.95" -> quantile 0.95
    "mean" -> mean
    "sum" -> sum

    todo: test this in the multi-variate setting yet.
    """

    def __init__(
        self,
        agg_window: int,
        agg_fns: List[str],
        rescale: bool = True,
        min_scale: float = 1.0e-12,
    ):
        self.agg_window = agg_window
        self.rescale = rescale
        self._agg = []
        for agf in agg_fns:
            m = re.match(r"q(\d\.\d+)", agf)
            if m:
                q = float(m.group(1))
                self._agg.append(
                    lambda x: torch.nanquantile(x, q=float(q), dim=-1)
                )
            elif agf == "mean":
                self._agg.append(
                    lambda x: torch.nansum(x, dim=-1) / x.shape[-1]
                )
            elif agf == "sum":
                self._agg.append(lambda x: torch.nansum(x, dim=-1))
            else:
                raise NotImplementedError()

        self.num_out_dim = len(self._agg)
        self.min_scale = min_scale

    def __call__(self, ts_batch):
        shape = ts_batch.shape
        assert len(shape) == 2
        T = shape[-1]
        assert (
            T % self.agg_window == 0
        ), f"{T} % {self.agg_window} = {T % self.agg_window}"

        ts_reshape = ts_batch.reshape((shape[0], -1, self.agg_window))
        res = [agf(ts_reshape) for agf in self._agg]
        res = torch.stack(res, dim=1)
        if self.rescale:
            scale = torch.nansum(torch.abs(ts_batch), dim=-1) / T
            scale = torch.clip(scale, self.min_scale, np.inf)
            res / scale[:, None, None]
        return res


class EmbedModel(pl.LightningModule):
    def __init__(
        self,
        channels: int,
        out_channels: int,
        depth: int,
        kernel_size: int,
        reduced_size: int,
        compared_length: Optional[int],
        lr: float,
        loss_temperature: float,
        multivar_dim: int,
        preprocessor: PreProcessor = None,
        **kwargs,
    ):
        """
        :param channels: Number of channels in convolutions (if it's >1, then we have a multi-variate time series)
        :param out_channels: Output dimension of encoder
        :param depth: Number of layers
        :param kernel_size: Convolution kernel size
        :param reduced_size:
        :param compared_length: Size of window used during training (this is the outer window that is selected)
        :param lr:
        :param loss_temperature:
        :param multivar_dim:
        :param preprocessor: defaults to ScalePreprocessor if nothing is selected
        :param kwargs:
        """
        super().__init__()
        self.save_hyperparameters()
        print(self.hparams)

        if preprocessor is None:
            self.preprocess = ScalePreProcessor(multivar_dim)
        else:
            self.preprocess = preprocessor

        in_channels = self.preprocess.num_out_dim

        self.encoder = causal_cnn.CausalCNNEncoder(
            in_channels=in_channels,
            channels=self.hparams.channels,
            depth=self.hparams.depth,
            out_channels=self.hparams.out_channels,
            kernel_size=self.hparams.kernel_size,
            reduced_size=self.hparams.reduced_size,
        )

        if self.hparams.loss == 'SimCLR':
            self.loss = contrastive_loss.NT_Xent_Loss(
                compared_length=self.hparams.compared_length,
                temperature=self.hparams.loss_temperature,
            )
        elif self.hparams.loss == 'BarlowTwins':
            self.loss = contrastive_loss.BarlowTwins(
                compared_length=self.hparams.compared_length,
                out_channels=self.hparams.out_channels,
                lambd=self.hparams.loss_lambda,
            )

    @staticmethod
    def add_model_specific_args(parent_parser):
        def parse_opt_tuple(x):
            if x is None or x == "":
                return None
            if isinstance(x, str):
                if not x:
                    return None
                return ast.literal_eval(x)
            return x

        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--channels", type=int, default=32)
        parser.add_argument("--out_channels", type=int, default=128)
        parser.add_argument("--depth", type=int, default=10)
        parser.add_argument("--kernel_size", type=int, default=3)
        parser.add_argument("--reduced_size", type=int, default=64)
        parser.add_argument("--compared_length", type=int, default=2 * 7 * 24)
        parser.add_argument("--lr", type=float, default=0.005)
        parser.add_argument("--loss_temperature", type=float, default=0.1)
        parser.add_argument("--loss_lambda", type=float, default=1e-2)
        parser.add_argument("--loss", type=str, default='SimCLR')
        return parser

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        u = self.preprocess(x)
        embedding = self.encoder(u)
        embedding = F.normalize(embedding, p=2, dim=-1)
        return embedding

    def training_step(self, batch, batch_idx):
        # TODO: If the pre-processing is expensive, we may optionally want to run it
        #       on the whole training dataset before training
        u = self.preprocess(batch)
        rep, sub_window_rep = self.loss.get_representations(u, self.encoder)
        rep = F.normalize(rep, p=2, dim=-1)
        sub_window_rep = F.normalize(sub_window_rep, p=2, dim=-1)
        loss = self.loss(rep, sub_window_rep)
        self.log(
            "hparam/train_loss",
            loss,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
        )
        # the loss can become NaN if we pick windows that only contain NaN values. We'll ignore these.
        if torch.any(loss.isnan()):
            return None
        else:
            return loss

    def on_epoch_end(self):
        print("\n")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        # we are using the lars optimizer which supposedly stabilizes training for large batch sizes
        optimizer = LARSWrapper(optimizer)
        return optimizer
