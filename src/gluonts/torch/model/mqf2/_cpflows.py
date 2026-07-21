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

# LICENSE: EXTERNAL
#
# This module vendors the minimal subset of CP-Flow required by the MQF2
# model, so that gluonts can provide the functionality without depending on
# the external ``cpflows`` package. It is adapted from the CP-Flow repository
# (https://github.com/CW-Huang/CP-Flow), which is distributed under the
# following license:
#
# MIT License
#
# Copyright (c) 2021 Chin-Wei Huang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import gc
import warnings
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# Distributions helper (from cpflows/distributions.py)
# ---------------------------------------------------------------------------

Log2PI = float(np.log(2 * np.pi))


def log_standard_normal(x):
    z = -0.5 * Log2PI
    return -(x**2) / 2 + z


# ---------------------------------------------------------------------------
# Log-determinant estimators (from cpflows/logdet_estimators.py)
# ---------------------------------------------------------------------------

EPS = 1e-7
CG_ITERS_TRACER = list()


def gram_schmidt_ortho(Q, v, tol=1e-5):
    """
    Orthogonalizes v wrt the rows vectors in Q.

    Assumes row vectors in Q are orthogonal and have unit Euclidean norm.

    Args:
        Q: (..., m, d)
        v: (..., d)
        tol: tolerance value for convergence
    Constraints:
        Q orthonormal, m < d.
    Returns:
        (..., d) Tensor that is orthogonal to all rows in Q.
    """
    *shape, m, d = Q.shape
    Q = Q.reshape(-1, m, d)
    assert shape == list(v.shape[:-1]), (
        f"Q and v need to have the same batch shape but"
        f" got Q.shape[:-2]={Q.shape[:-2]} and v.shape[:-1]={v.shape[:-1]}"
    )
    v = v.reshape(-1, d)

    # Check Q is orthonormal
    with torch.no_grad():
        if m > 1:
            QQ = torch.einsum("bmd,bnd->bmn", Q, Q)
            diagmask = torch.eye(QQ.shape[-1])[None].bool()
            offdiags = QQ[~diagmask.expand_as(QQ)].reshape(-1, m, m - 1)
            if (offdiags > tol).any():
                print("Warning: non-orthogonal rows in Q")

    inner_qv = torch.einsum("bmd,bd->bm", Q, v)
    proj_v = torch.einsum("bm,bmd->bd", inner_qv, Q)
    v = v - proj_v

    # EPS is added here so we can divide and multiply by the same number later.
    v_norm = torch.norm(v, dim=-1, keepdim=True).detach() + EPS

    # Verify orthogonality
    inner_qv = torch.einsum("bmd,bd->bm", Q, v / v_norm)
    retries = 0
    while (torch.abs(inner_qv) > tol).any():
        proj_v = torch.einsum("bm,bmd->bd", inner_qv * v_norm, Q)
        v = v - proj_v
        inner_qv = torch.einsum("bmd,bd->bm", Q, v)
        retries += 1
        if retries >= 10:
            print("Warning: orthogonalization exceeded 10 retries.")
            break

    return v.reshape(*shape, d)


def lanczos_tridiagonalization(hvp_fun, m, v):
    """
    Args:
        hvp_fun: A broadcastable function that computes Hessian-vector products.
        m: number of Lanczos iterations.
        v: (bsz, d) a starting orthonormal vector.
    Returns:
        A tridiagonal matrix (m, m) resulting from the Lanczos method.
    """
    bsz, d = v.shape

    # Multiple torch.stack; need better implementation.
    vecs = [v]
    Q = torch.stack(vecs, dim=1)

    w = hvp_fun(v)
    alpha = torch.einsum("bi,bi->b", w, v)
    w = gram_schmidt_ortho(Q, w)

    alphas = [alpha]
    betas = []

    for j in range(2, m + 1):
        beta = div = torch.norm(w, dim=-1)

        while (div < EPS).any():
            # TODO: only a couple batches are small; more memory efficient?
            print(f"rerolling {(div < EPS).sum().item()} vectors")
            idx = beta < EPS
            w_new = torch.nn.functional.normalize(torch.randn_like(w), dim=-1)
            w_new = gram_schmidt_ortho(Q, w_new)
            w = w * ~idx.unsqueeze(-1) + w_new * idx.unsqueeze(-1)
            div = torch.norm(w, dim=-1)

        v = F.normalize(w, dim=-1)
        vecs.append(v)

        Q = torch.stack(vecs, dim=1)

        w = hvp_fun(v)
        alpha = torch.einsum("bi,bi->b", w, v)
        w = gram_schmidt_ortho(Q, w)

        alphas.append(alpha)
        betas.append(beta)

    alphas = torch.stack(alphas, dim=-1)
    betas = torch.stack(betas, dim=-1)

    T = (
        torch.diag_embed(betas, offset=-1)
        + torch.diag_embed(alphas, offset=0)
        + torch.diag_embed(betas, offset=1)
    )
    return T


def stochastic_quadrature(T, dim, func=torch.log):
    # ``torch.symeig`` was removed in torch 2.0; ``torch.linalg.eigh`` is the
    # modern equivalent and returns eigenvalues in ascending order.
    eigvals, eigvecs = torch.linalg.eigh(T)

    with torch.no_grad():
        if eigvals.numel() > torch.unique(eigvals, dim=1).numel():
            print("Non-unique eigenvalues.")

    clamped_eigvals = eigvals.clamp(min=0).detach() + (
        eigvals - eigvals.detach()
    )
    tau = eigvecs[..., 0, :]
    return torch.sum(tau * tau * func(clamped_eigvals + 1e-8), dim=-1) * dim


def stochastic_lanczos_quadrature(hvp_fun, v, m, func=torch.log):
    bsz, dim = v.shape
    T = lanczos_tridiagonalization(hvp_fun, m, v)
    return stochastic_quadrature(T, dim, func=func)


def batch_dot_product(a, b):
    return torch.bmm(a.unsqueeze(1), b.unsqueeze(2)).squeeze(2)


def conjugate_gradient(hvp, b, m=10, rtol=0.0, atol=1e-3):
    """
    Solves H^{-1} v using m iterations of conjugate gradient.

    v is (bsz, dim) and output shape should be (bsz, dim).
    """
    # initialization
    # could also initialize other ways, e.g. `x = torch.ones_like(b)`
    x = b.clone().detach()
    r = b - hvp(x)
    tol = atol + rtol * torch.abs(x)
    if (torch.abs(r) < tol).all():
        CG_ITERS_TRACER.append(0)
        return x
    p = r
    r2 = batch_dot_product(r, r)
    k = 0
    while k < m:
        k += 1
        Ap = hvp(p)

        a = r2 / (batch_dot_product(p, Ap) + 1e-8)
        x = x + a * p
        r = r - a * Ap
        tol = atol + rtol * torch.abs(x)
        if (torch.abs(r) < tol).all():
            break
        r2_new = batch_dot_product(r, r)
        beta = r2_new / r2
        r2 = r2_new
        p = r + beta * p
    CG_ITERS_TRACER.append(k)
    return x


def stochastic_logdet_gradient_estimator(hvp_fun, v, m, rtol=0.0, atol=1e-3):
    with torch.no_grad():
        v_Hinv = conjugate_gradient(hvp_fun, v, m, rtol=rtol, atol=atol)
    surrog_logdet = torch.sum(hvp_fun(v_Hinv) * v, dim=1)
    return surrog_logdet


def unbiased_logdet(hvp_fun, v, p=0.1, n_exact_terms=4):
    bsz, dim = v.shape
    m = geometric_sample(p) + n_exact_terms

    def coeff_fn(kk):
        return 1 / geometric_1mcdf(p, kk, n_exact_terms)

    T = lanczos_tridiagonalization(hvp_fun, m, v)
    estimate = prev_estimate = 0.0
    for k in range(n_exact_terms, m + 1):
        logdet_estimate = stochastic_quadrature(T[:, :k, :k], dim)
        estimate = estimate + coeff_fn(k) * (logdet_estimate - prev_estimate)
        prev_estimate = logdet_estimate
    return estimate


def geometric_sample(p):
    return np.random.geometric(p)


def geometric_1mcdf(p, k, offset=0):
    if k <= offset:
        return 1.0
    else:
        k = k - offset
    # P(n >= k)
    return (1 - p) ** max(k - 1, 0)


# ---------------------------------------------------------------------------
# Flows (from cpflows/flows/flows.py)
# ---------------------------------------------------------------------------

_scaling_min = 0.001


class ActNorm(torch.nn.Module):
    """ActNorm layer with data-dependant init."""

    def __init__(
        self, num_features, logscale_factor=1.0, scale=1.0, learn_scale=True
    ):
        super(ActNorm, self).__init__()
        self.initialized = False
        self.num_features = num_features

        self.register_parameter(
            "b",
            nn.Parameter(torch.zeros(1, num_features, 1), requires_grad=True),
        )
        self.learn_scale = learn_scale
        if learn_scale:
            self.logscale_factor = logscale_factor
            self.scale = scale
            self.register_parameter(
                "logs",
                nn.Parameter(
                    torch.zeros(1, num_features, 1), requires_grad=True
                ),
            )

    def forward_transform(self, x, logdet=0):
        input_shape = x.size()
        x = x.view(input_shape[0], input_shape[1], -1)

        if not self.initialized:
            self.initialized = True

            def unsqueeze(x):
                return x.unsqueeze(0).unsqueeze(-1).detach()

            # Compute the mean and variance
            sum_size = x.size(0) * x.size(-1)
            b = -torch.sum(x, dim=(0, -1)) / sum_size
            self.b.data.copy_(unsqueeze(b).data)

            if self.learn_scale:
                var = unsqueeze(
                    torch.sum((x + unsqueeze(b)) ** 2, dim=(0, -1)) / sum_size
                )
                logs = (
                    torch.log(self.scale / (torch.sqrt(var) + 1e-6))
                    / self.logscale_factor
                )
                self.logs.data.copy_(logs.data)

        b = self.b
        output = x + b

        if self.learn_scale:
            logs = self.logs * self.logscale_factor
            scale = torch.exp(logs) + _scaling_min
            output = output * scale
            dlogdet = torch.sum(torch.log(scale)) * x.size(-1)  # c x h

            return output.view(input_shape), logdet + dlogdet
        else:
            return output.view(input_shape), logdet

    def reverse(self, y, **kwargs):
        assert self.initialized
        input_shape = y.size()
        y = y.view(input_shape[0], input_shape[1], -1)
        logs = self.logs * self.logscale_factor
        b = self.b
        scale = torch.exp(logs) + _scaling_min
        x = y / scale - b

        return x.view(input_shape)

    def extra_repr(self):
        return f"{self.num_features}"


class ActNormNoLogdet(ActNorm):
    def forward(self, x):
        return super(ActNormNoLogdet, self).forward_transform(x)[0]


class SequentialFlow(torch.nn.Module):
    def __init__(self, flows):
        super(SequentialFlow, self).__init__()
        self.flows = torch.nn.ModuleList(flows)

    def forward_transform(self, x, logdet=0, context=None, extra=None):
        for flow in self.flows:
            if isinstance(flow, DeepConvexFlow):
                x, logdet = flow.forward_transform(
                    x, logdet, context=context, extra=extra
                )
            else:
                prev_logdet = logdet
                x, logdet = flow.forward_transform(x, logdet)
                if extra is not None and len(extra) > 0:
                    extra[0] = extra[0] + (logdet - prev_logdet).detach()
        return x, logdet

    def reverse(self, x, **kwargs):
        for flow in self.flows[::-1]:
            x = flow.reverse(x, **kwargs)
        return x

    def logp(self, x, context=None, extra=None):
        z, logdet = self.forward_transform(x, context=context, extra=extra)
        logp0 = log_standard_normal(z).sum(-1)
        if extra is not None and len(extra) > 0:
            extra[0] = extra[0] + logp0.detach()
        return logp0 + logdet


# ---------------------------------------------------------------------------
# Convex potential flows (from cpflows/flows/cpflows.py)
# ---------------------------------------------------------------------------

HESS_NORM_TRACER = list()


class DeepConvexFlow(torch.nn.Module):
    """
    Deep convex potential flow parameterized by an input-convex neural network.

    This is the main framework used in the paper.
    The ``forward_transform_stochastic`` function is used to give a stochastic
    estimate of the logdet "gradient" during training, and a stochastic
    estimate of the logdet itself on eval mode (using Lanczos).
    The ``forward_transform_bruteforce`` function computes the logdet exactly.
    """

    def __init__(
        self,
        icnn,
        dim,
        unbiased=False,
        no_bruteforce=True,
        m1=10,
        m2=None,
        rtol=0.0,
        atol=1e-3,
        bias_w1=0.0,
        trainable_w0=True,
    ):
        super(DeepConvexFlow, self).__init__()
        if m2 is None:
            m2 = dim
        self.icnn = icnn
        self.no_bruteforce = no_bruteforce
        self.rtol = rtol
        self.atol = atol

        self.w0 = torch.nn.Parameter(
            torch.log(torch.exp(torch.ones(1)) - 1),
            requires_grad=trainable_w0,
        )
        self.w1 = torch.nn.Parameter(torch.zeros(1) + bias_w1)
        self.bias_w1 = bias_w1

        self.m1, self.m2 = m1, m2
        self.stochastic_estimate_fn = (
            unbiased_logdet
            if unbiased
            else partial(stochastic_lanczos_quadrature, m=min(m1, dim))
        )
        self.stochastic_grad_estimate_fn = partial(
            stochastic_logdet_gradient_estimator,
            m=min(m2, dim),
            rtol=self.rtol,
            atol=self.atol,
        )

    def get_potential(self, x, context=None):
        n = x.size(0)
        if context is None:
            icnn = self.icnn(x)
        else:
            icnn = self.icnn(x, context)
        return (
            F.softplus(self.w1) * icnn
            + F.softplus(self.w0)
            * (x.view(n, -1) ** 2).sum(1, keepdim=True)
            / 2
        )

    def reverse(
        self,
        y,
        max_iter=1000000,
        lr=1.0,
        tol=1e-12,
        x=None,
        context=None,
        **kwargs,
    ):
        if x is None:
            x = y.clone().detach().requires_grad_(True)

        def closure():
            # Solves x such that f(x) - y = 0
            # <=> Solves x such that argmin_x F(x) - <x,y>
            F_ = self.get_potential(x, context)
            loss = torch.sum(F_) - torch.sum(x * y)
            x.grad = torch.autograd.grad(loss, x)[0].detach()
            return loss

        optimizer = torch.optim.LBFGS(
            [x],
            lr=lr,
            line_search_fn="strong_wolfe",
            max_iter=max_iter,
            tolerance_grad=tol,
            tolerance_change=tol,
        )

        optimizer.step(closure)

        torch.cuda.empty_cache()
        gc.collect()

        return x

    def forward(self, x, context=None):
        with torch.enable_grad():
            x = x.clone().requires_grad_(True)
            F_ = self.get_potential(x, context)
            f = torch.autograd.grad(F_.sum(), x, create_graph=True)[0]
        return f

    def forward_transform(self, x, logdet=0, context=None, extra=None):
        if self.training or self.no_bruteforce:
            return self.forward_transform_stochastic(
                x, logdet, context=context, extra=extra
            )
        else:
            return self.forward_transform_bruteforce(
                x, logdet, context=context
            )

    def forward_transform_stochastic(
        self, x, logdet=0, context=None, extra=None
    ):
        bsz, *dims = x.shape
        dim = np.prod(dims)

        with torch.enable_grad():
            x = x.clone().requires_grad_(True)

            F_ = self.get_potential(x, context)
            f = torch.autograd.grad(F_.sum(), x, create_graph=True)[0]

            def hvp_fun(v):
                # v is (bsz, dim)
                v = v.reshape(bsz, *dims)
                hvp = torch.autograd.grad(
                    f, x, v, create_graph=self.training, retain_graph=True
                )[0]

                HESS_NORM_TRACER.append(
                    (torch.norm(hvp) / torch.norm(v)).detach().cpu()
                )

                if not torch.isnan(v).any() and torch.isnan(hvp).any():
                    raise ArithmeticError("v has no nans but hvp has nans.")
                hvp = hvp.reshape(bsz, dim)
                return hvp

        if self.training:
            v1 = sample_rademacher(bsz, dim).to(x)
            est1 = self.stochastic_grad_estimate_fn(hvp_fun, v1)
        else:
            est1 = 0

        if not self.training or (extra is not None and len(extra) > 0):
            try:
                v2 = torch.nn.functional.normalize(
                    sample_rademacher(bsz, dim), dim=-1
                ).to(f)
                est2 = self.stochastic_estimate_fn(hvp_fun, v2)
            except Exception:
                import traceback

                print(
                    "stochastic_estimate_fn failed with the following error"
                    " message:"
                )
                print(traceback.format_exc(), flush=True)
                est2 = torch.zeros_like(logdet).fill_(float("nan"))
            if extra is not None and len(extra) > 0:
                extra[0] = extra[0] + est2.detach()
        else:
            est2 = 0

        return f, logdet + est1 if self.training else logdet + est2

    def forward_transform_bruteforce(self, x, logdet=0, context=None):
        warnings.warn("brute force")
        bsz = x.shape[0]
        input_shape = x.shape[1:]

        with torch.enable_grad():
            x.requires_grad_(True)
            F_ = self.get_potential(x, context)
            f = torch.autograd.grad(F_.sum(), x, create_graph=True)[0]

            # TODO: compute Hessian in block mode instead of row-by-row.
            f = f.reshape(bsz, -1)
            H = []
            for i in range(f.shape[1]):
                retain_graph = self.training or (i < (f.shape[1] - 1))
                H.append(
                    torch.autograd.grad(
                        f[:, i].sum(),
                        x,
                        create_graph=self.training,
                        retain_graph=retain_graph,
                    )[0]
                )

            # H is (bsz, dim, dim)
            H = torch.stack(H, dim=1)

        f = f.reshape(bsz, *input_shape)
        return f, logdet + torch.slogdet(H).logabsdet

    def extra_repr(self):
        return f"ConjGrad(rtol={self.rtol}, atol={self.atol})"


def sample_rademacher(*shape):
    return (torch.rand(*shape) > 0.5).float() * 2 - 1


# ---------------------------------------------------------------------------
# Input-convex neural networks (from cpflows/icnn.py)
# ---------------------------------------------------------------------------


def symm_softplus(x, softplus_=torch.nn.functional.softplus):
    return softplus_(x) - 0.5 * x


def softplus(x):
    return nn.functional.softplus(x)


def gaussian_softplus(x):
    z = np.sqrt(np.pi / 2)
    return (
        z * x * torch.erf(x / np.sqrt(2)) + torch.exp(-(x**2) / 2) + z * x
    ) / (2 * z)


def gaussian_softplus2(x):
    z = np.sqrt(np.pi / 2)
    return (
        z * x * torch.erf(x / np.sqrt(2)) + torch.exp(-(x**2) / 2) + z * x
    ) / z


def laplace_softplus(x):
    return torch.relu(x) + torch.exp(-torch.abs(x)) / 2


def cauchy_softplus(x):
    # (Pi y + 2 y ArcTan[y] - Log[1 + y ^ 2]) / (2 Pi)
    pi = np.pi
    return (x * pi - torch.log(x**2 + 1) + 2 * x * torch.atan(x)) / (2 * pi)


def activation_shifting(activation):
    def shifted_activation(x):
        return activation(x) - activation(torch.zeros_like(x))

    return shifted_activation


def get_softplus(softplus_type="softplus", zero_softplus=False):
    if softplus_type == "softplus":
        act = nn.functional.softplus
    elif softplus_type == "gaussian_softplus":
        act = gaussian_softplus
    elif softplus_type == "gaussian_softplus2":
        act = gaussian_softplus2
    elif softplus_type == "laplace_softplus":
        act = gaussian_softplus
    elif softplus_type == "cauchy_softplus":
        act = cauchy_softplus
    else:
        raise NotImplementedError(
            f"softplus type {softplus_type} not supported."
        )
    if zero_softplus:
        act = activation_shifting(act)
    return act


class Softplus(nn.Module):
    def __init__(self, softplus_type="softplus", zero_softplus=False):
        super(Softplus, self).__init__()
        self.softplus_type = softplus_type
        self.zero_softplus = zero_softplus

    def forward(self, x):
        return get_softplus(self.softplus_type, self.zero_softplus)(x)


class PosLinear(torch.nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        gain = 1 / x.size(1)
        return (
            nn.functional.linear(
                x, torch.nn.functional.softplus(self.weight), self.bias
            )
            * gain
        )


class PICNN(torch.nn.Module):
    def __init__(
        self,
        dim=2,
        dimh=16,
        dimc=2,
        num_hidden_layers=2,
        PosLin=PosLinear,
        symm_act_first=False,
        softplus_type="gaussian_softplus",
        zero_softplus=False,
    ):
        super(PICNN, self).__init__()
        # with data dependent init

        self.act = Softplus(
            softplus_type=softplus_type, zero_softplus=zero_softplus
        )
        self.act_c = nn.ELU()
        self.symm_act_first = symm_act_first

        # data path
        Wzs = list()
        Wzs.append(nn.Linear(dim, dimh))
        for _ in range(num_hidden_layers - 1):
            Wzs.append(PosLin(dimh, dimh, bias=True))
        Wzs.append(PosLin(dimh, 1, bias=False))
        self.Wzs = torch.nn.ModuleList(Wzs)

        # skip data
        Wxs = list()
        for _ in range(num_hidden_layers - 1):
            Wxs.append(nn.Linear(dim, dimh))
        Wxs.append(nn.Linear(dim, 1, bias=False))
        self.Wxs = torch.nn.ModuleList(Wxs)

        # context path
        Wcs = list()
        Wcs.append(nn.Linear(dimc, dimh))
        self.Wcs = torch.nn.ModuleList(Wcs)

        Wczs = list()
        for _ in range(num_hidden_layers - 1):
            Wczs.append(nn.Linear(dimh, dimh))
        Wczs.append(nn.Linear(dimh, dimh, bias=True))
        self.Wczs = torch.nn.ModuleList(Wczs)
        for Wcz in self.Wczs:
            Wcz.weight.data.zero_()
            Wcz.bias.data.zero_()

        Wcxs = list()
        for _ in range(num_hidden_layers - 1):
            Wcxs.append(nn.Linear(dimh, dim))
        Wcxs.append(nn.Linear(dimh, dim, bias=True))
        self.Wcxs = torch.nn.ModuleList(Wcxs)
        for Wcx in self.Wcxs:
            Wcx.weight.data.zero_()
            Wcx.bias.data.zero_()

        Wccs = list()
        for _ in range(num_hidden_layers - 1):
            Wccs.append(nn.Linear(dimh, dimh))
        self.Wccs = torch.nn.ModuleList(Wccs)

        self.actnorm0 = ActNormNoLogdet(dimh)
        actnorms = list()
        for _ in range(num_hidden_layers - 1):
            actnorms.append(ActNormNoLogdet(dimh))
        actnorms.append(ActNormNoLogdet(1))
        self.actnorms = torch.nn.ModuleList(actnorms)

        self.actnormc = ActNormNoLogdet(dimh)

    def forward(self, x, c):
        if self.symm_act_first:
            z = symm_softplus(self.actnorm0(self.Wzs[0](x)), self.act)
        else:
            z = self.act(self.actnorm0(self.Wzs[0](x)))
        c = self.act_c(self.actnormc(self.Wcs[0](c)))
        for Wz, Wx, Wcz, Wcx, Wcc, actnorm in zip(
            self.Wzs[1:-1],
            self.Wxs[:-1],
            self.Wczs[:-1],
            self.Wcxs[:-1],
            self.Wccs,
            self.actnorms[:-1],
        ):
            cz = softplus(Wcz(c) + np.exp(np.log(1.0) - 1))
            cx = Wcx(c) + 1.0
            z = self.act(actnorm(Wz(z * cz) + Wx(x * cx) + Wcc(c)))

        cz = softplus(self.Wczs[-1](c) + np.log(np.exp(1.0) - 1))
        cx = self.Wcxs[-1](c) + 1.0
        return self.actnorms[-1](self.Wzs[-1](z * cz) + self.Wxs[-1](x * cx))
