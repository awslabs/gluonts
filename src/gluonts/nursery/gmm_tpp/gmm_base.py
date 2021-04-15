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
import scipy as sp
import pylab as pl
import time
import mxnet as mx
from mxnet import gluon


class GMMModel(gluon.HybridBlock):
    """
    log p(x) >= E_{q(z|x)}[ log p(x|z) + log p(z) - log q(z|x) ]
    p(z) = unif
    p(x|z) = N(x; mu_z, (kR_z'kR_z)^-1)
    q(z|x) = softmax(-0.5*(kR_z(x-mu_z))^2 +logdet(kR_z) -0.5d*log(2*pi))
    mu_z = E_{q(z|x)} x
    cov_z = E_{q(z|x)} x^2 - (E_{q(z|x)} x)^2
    kR_z = inv(choL(cov_z))

    Shapes
    ===
    x: (batch_size, input_dim)
    log_marg: (batch_size,)
    qz: (batch_size, num_clusters)
    """

    def __init__(
        self,
        *data_template,
        num_clusters,
        log_prior_=None,
        mu_=None,
        kR_=None,
        lr_mult=None,
        ctx=None,
        hybridize=None,
    ):
        super().__init__()
        self.input_dim = data_template[0].shape[1]
        self.num_clusters = num_clusters

        with self.name_scope():
            self.log_prior_ = self.params.get(
                "log_prior_",
                shape=(num_clusters,),
                lr_mult=lr_mult,
                init=mx.init.Constant(np.log(1 / self.num_clusters))
                if log_prior_ is None
                else mx.init.Constant(log_prior_),
            )

            self.mu_ = self.params.get(
                "mu_",
                shape=(self.num_clusters, self.input_dim),
                lr_mult=lr_mult,
                init=None if mu_ is None else mx.init.Constant(mu_),
            )

            self.kR_ = self.params.get(
                "kR_",
                shape=(self.num_clusters, self.input_dim, self.input_dim),
                lr_mult=lr_mult,
                init=None if kR_ is None else mx.init.Constant(kR_),
            )

        if hybridize:
            self.hybridize()

        self.initialize(ctx=ctx)
        self(*[mx.nd.array(x, ctx=ctx) for x in data_template])

    @staticmethod
    def _get_dx_(F, x, mu_):
        """ @Return (batch_size, num_clusters, input_dim) """
        return F.broadcast_minus(x.expand_dims(1), mu_)

    @staticmethod
    def _get_Rx_(F, dx_, kR_):
        """ @Return (batch_size, num_clusters, input_dim) """
        kR_expand_0 = F.broadcast_like(
            kR_.expand_dims(0), dx_, lhs_axes=(0,), rhs_axes=(0,)
        )  # (batch_size, num_clusters, input_dim, input_dim)

        Rx_ = F.batch_dot(kR_expand_0, dx_.expand_dims(-1)).squeeze(axis=-1)
        return Rx_

    def hybrid_forward(self, F, x, log_prior_, mu_, kR_):
        """ E-step computes log_marginal and q(z|x) """
        dx_ = self._get_dx_(F, x, mu_)
        Rx_ = self._get_Rx_(F, dx_, kR_)

        log_conditional = (
            -0.5 * (Rx_ ** 2).sum(axis=-1)
            - 0.5 * self.input_dim * np.log(2 * np.pi)
            + F.linalg.slogdet(kR_)[1]
        )  # (batch, num_clusters)

        log_complete = F.broadcast_add(
            log_conditional, log_prior_.log_softmax()
        )
        log_incomplete = F.log(F.exp(log_complete).sum(axis=1))

        qz = log_complete.softmax(axis=1)

        return log_incomplete, qz

    @staticmethod
    def m_step(x, qz):
        """ M-step computes summary statistics in numpy """
        x = x.astype("float64")
        qz = qz.astype("float64")

        nz = qz.sum(axis=0)  # (num_clusters,)
        sum_x = (qz[:, :, None] * x[:, None, :]).sum(axis=0)
        sum_x2 = (
            qz[:, :, None, None] * (x[:, None, :, None] @ x[:, None, None, :])
        ).sum(axis=0)
        return nz, sum_x, sum_x2


class GMMTrainer:
    """trainer based on M-step summary statistics
    can add mini-batch statistics for a full-batch update
    """

    def __init__(self, model, pseudo_count=0.1, jitter=1e-6):
        self.model = model
        self.pseudo_count = pseudo_count
        self.jitter = jitter
        self.zero_stats()

    def zero_stats(self):
        self.nz = self.pseudo_count * np.ones(self.model.num_clusters)
        self.sum_x = np.zeros(self.model.input_dim)
        self.sum_x2 = (
            np.eye(self.model.input_dim) * self.jitter * self.pseudo_count
        )

    def add(self, x):
        log_incomplete, qz = self.model(mx.nd.array(x))
        nz, sum_x, sum_x2 = self.model.m_step(x, qz.asnumpy())
        self.nz = self.nz + nz
        self.sum_x = self.sum_x + sum_x
        self.sum_x2 = self.sum_x2 + sum_x2
        return log_incomplete

    def update(self):
        mu_ = self.sum_x / self.nz[:, None]
        Ex2 = self.sum_x2 / self.nz[:, None, None]
        cov_ = Ex2 - mu_[:, :, None] @ mu_[:, None, :]
        kR_ = np.linalg.inv(np.linalg.cholesky(cov_))

        self.model.log_prior_.set_data(np.log(self.nz / self.nz.sum()))
        self.model.mu_.set_data(mu_)
        self.model.kR_.set_data(kR_)

        self.zero_stats()

    def __call__(self, x):
        self.add(x)
        self.update()


def infer_lambda(model, *_, xmin, xmax):
    """ infer lambda and intercept based on linear fitting at the base points """
    x = np.linspace(xmin, xmax).reshape((-1, 1))
    y = np.ravel(model(mx.nd.array(x))[0].asnumpy())
    slope, intercept = np.polyfit(np.ravel(x), np.ravel(y), 1)
    return -slope


def elapsed(collection):
    """ similar to enumerate but prepend elapsed time since loop starts """
    tic = time.time()
    for x in collection:
        yield time.time() - tic, x
