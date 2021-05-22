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


import numpy as np
import mxnet as mx
from mxnet import gluon, nd, autograd
import matplotlib.pyplot as plt
from mxnet.gluon import nn, rnn
from tqdm import tqdm_notebook as tqdm
import math


class NeuralLV:
    def __init__(
        self,
        num_time_series,
        num_time_steps,
        low_rank_param,
        is_full_matrix,
        p0,
        r,
        k,
        A,
        is_sym,
    ):
        # Define number of time series
        self.num_time_series = num_time_series
        # Define number of discrete time steps will assume the same for each time series so p_i(t) in R^(dxN)
        self.num_time_steps = num_time_steps
        self.low_rank_param = low_rank_param
        self.is_full_matrix = is_full_matrix
        self.p0 = p0
        self.r = r
        self.k = k
        self.A = A
        self.is_sym = is_sym
        self.dtype = r.dtype
        self.ctx = r.context

    # mat1 = A is the default case.  If low_rank is on, we have the symmetric case where A=B^TB
    # and mat1 = B and in the non-symmetric case we need to pass in mat2 = C so A = B^TC
    def solve_discrete_lv(self, mat1, mat2=None, is_full_matrix=True):
        p = (
            []
        )  # need to store as list for autograd won't let you append indices in same matrix
        p.append(self.p0)
        for n in range(
            self.num_time_steps - 1
        ):  # element-wise vector division and multiplication
            # Compute Ap to generate synthetic data for the full rank matrix A
            if is_full_matrix:
                mat_vec_prod = nd.dot(mat1, p[n])
            else:
                mat_vec_prod = compute_mat_vec_prod(mat1, mat2, p[n])
            p.append((1 + self.r * (1 - mat_vec_prod / self.k)) * p[n])
            # concat puts in nd array of size num_ts*N
            # need to take size (N, num_ts) and transpose otherwise default is doing row major
            # and we need column major storing of the linear list p
        return (
            nd.concat(*p, dim=0)
            .reshape(self.num_time_steps, self.num_time_series)
            .T
        )

    # p0: initial condition shape (num_ts, )
    # r: growth rate shape (num_ts, ) TODO: Learn
    # k: carrying capacity shape (num_ts, ) TODO: Learn
    # A: interaction matrix shape (num_ts, num_ts)
    def run(self, num_epochs=1000, model=None):
        # Compute exact solution with full rank matrix
        p = self.solve_discrete_lv(self.A)
        # To initialize model otherwise can feed in model and rerun
        if model is None:
            model = LowRankVectorEmbedding(self)
            model.collect_params().initialize(
                mx.init.Xavier(magnitude=2.24), force_reinit=True, ctx=self.ctx
            )
        trainer = gluon.Trainer(model.collect_params(), "adam")
        p_approx = train(p, trainer, model, num_epochs)
        B = model.embedding_low_rank_mat_B(model.feat_static_cat).T
        C = (
            None
            if self.is_sym
            else model.embedding_low_rank_mat_C(model.feat_static_cat).T
        )
        A_approx = compute_low_rank_product(B, C)
        return p_approx, p, A_approx, model


class LowRankVectorEmbedding(gluon.HybridBlock):
    def __init__(self, neural_lv, **kwargs):
        super().__init__(**kwargs)
        self.num_time_series = neural_lv.num_time_series  # num_ts
        self.low_rank_param = neural_lv.low_rank_param  # low rank parameter k
        self.is_full_matrix = neural_lv.is_full_matrix
        self.dtype = neural_lv.dtype
        self.ctx = neural_lv.ctx
        self.feat_static_cat = nd.arange(
            self.num_time_series, ctx=self.ctx, dtype=self.dtype
        )
        self.is_sym = neural_lv.is_sym
        self.neural_lv = neural_lv

        with self.name_scope():
            self.embedding_low_rank_mat_B = gluon.nn.Embedding(
                input_dim=self.num_time_series,
                output_dim=self.low_rank_param,
                dtype=self.dtype,
            )
            if not self.is_sym:
                self.embedding_low_rank_mat_C = gluon.nn.Embedding(
                    input_dim=self.num_time_series,
                    output_dim=self.low_rank_param,
                    dtype=self.dtype,
                )

    def forward(self):
        # find low rank vector computed per time series
        # feat_static_cat consists of the time series indices 0, ..., cardinality - 1
        # embedding returns (num_ts, low_rank_param) need to transpose it
        B = self.embedding_low_rank_mat_B(self.feat_static_cat).T
        C = (
            None
            if self.is_sym
            else self.embedding_low_rank_mat_C(self.feat_static_cat).T
        )
        # Explicitly form matrix matrix product A = B^T* CO(kd^2) expensive
        if self.is_full_matrix:
            return self.neural_lv.solve_discrete_lv(
                compute_low_rank_product(B, C)
            )
        # Compute matrix vector product B^T * (C p) O(kd)
        else:
            return self.neural_lv.solve_discrete_lv(B, C, self.is_full_matrix)


def train(p, trainer, model, num_epochs=1000):
    loss = gluon.loss.L2Loss()
    tqdm_epochs = tqdm(range(num_epochs))
    for e in tqdm_epochs:
        with autograd.record():
            # forward pass
            p_approx = model.forward()
            Loss = loss(p, p_approx)
            Loss.backward()
        trainer.step(model.num_time_series)
        tqdm_epochs.set_postfix({"loss": nd.sum(Loss).asscalar()})
    return p_approx


def lv_plot_ts(
    p, p_approx, max_num_plots=10, num_rows=2, fig_size_width=10
):  # plots all time series at corresponding time point
    plt.rcParams["figure.figsize"] = (fig_size_width, 5)
    num_ts = p.shape[0]
    N = p.shape[1]
    t = np.arange(N)
    num_plots = min(num_ts, max_num_plots)
    num_cols = int(num_plots / num_rows)
    fig, axs = plt.subplots(num_rows, num_cols)
    for ts_idx in range(num_plots):
        plt.subplot(num_rows, num_cols, ts_idx + 1)
        plt.plot(
            t, p[ts_idx, :].asnumpy(), t, p_approx[ts_idx, :].asnumpy(), "r--"
        )
        plt.ylabel(f"$p_{ts_idx}(t)$")
        plt.xlabel("t")
        plt.legend(("Exact", "Approx"))
        plt.xlabel("time: $t$")
        plt.ylabel(f"$p_{ts_idx}(t)$")


# Compute B^Tz, where z = C*p, C = B in the symmetic case
def compute_mat_vec_prod(B, C, p):
    # Can replace dot with nd.linalg.gemm2 for the matrix vector multiplication
    z = nd.dot(C, p) if C is not None else nd.dot(B, p)
    return nd.dot(B, z, transpose_a=True)


# A = B^TC, where C = B in the symmetric case
def compute_low_rank_product(B, C):
    return (
        nd.dot(B, C, transpose_a=True)
        if C is not None
        else nd.dot(B, B, transpose_a=True)
    )


# Returns random samples fromuniform distribution [0,1] can change to randn for normally distributed random values
def generate_data(
    num_ts, ctx=mx.gpu(), dtype="float64", seed=100
):  # num_ts = d
    np.random.seed(seed)
    # vector of shape (num_ts, )
    r = np.random.rand(num_ts)
    # vector of shape (num_ts, )
    k = np.random.rand(num_ts)
    # matrix of shape (num_ts, num_ts)
    A = np.random.rand(num_ts, num_ts)
    # diagonal entries are 1 representing intraspecies competition
    np.fill_diagonal(A, 1)
    # initial condition vector of shape (num_ts, )
    p0 = np.random.rand(num_ts)
    return (
        nd.array(r, ctx=ctx, dtype=dtype),
        nd.array(k, ctx=ctx, dtype=dtype),
        nd.array(p0, ctx=ctx, dtype=dtype),
        nd.array(A, ctx=ctx, dtype=dtype),
    )
