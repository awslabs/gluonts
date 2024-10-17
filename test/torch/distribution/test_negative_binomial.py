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

import pytest
import sys

import numpy as np
import torch

from gluonts.torch.distributions.negative_binomial import NegativeBinomial


@pytest.mark.parametrize("probs", [0.05, 0.5, 0.8])
@pytest.mark.parametrize("total_count", [3, 7, 100])
def test_custom_neg_bin_logpdf_matches_scipy_neg_bin_logpdf(
    total_count, probs
):
    torch_dist = NegativeBinomial(total_count=total_count, probs=probs)
    scipy_dist = torch_dist.scipy_nbinom
    x = torch.randint(1, 20, (1000,))
    log_pdf_torch = torch_dist.log_prob(x).numpy()
    log_pdf_scipy = scipy_dist.logpmf(x.numpy())

    assert np.allclose(log_pdf_torch, log_pdf_scipy, rtol=1e-4)


@pytest.mark.parametrize("probs", [0.1, 0.5, 0.8])
@pytest.mark.parametrize("total_count", [3, 7, 100])
@pytest.mark.parametrize("value", [11.0, 5.0, 9.0])
def test_custom_neg_bin_cdf(total_count, probs, value):
    torch_dist = NegativeBinomial(total_count=total_count, probs=probs)
    scipy_dist = torch_dist.scipy_nbinom

    torch_cdf = torch_dist.cdf(torch.as_tensor(value)).numpy()
    scipy_cdf = scipy_dist.cdf(np.asarray(value))

    assert np.allclose(torch_cdf, scipy_cdf)


@pytest.mark.parametrize("probs", [0.1, 0.5, 0.8])
@pytest.mark.parametrize("total_count", [3, 7, 100])
@pytest.mark.parametrize("value", [0.1, 0.5, 0.9])
def test_custom_neg_bin_icdf(total_count, probs, value):
    torch_dist = NegativeBinomial(total_count=total_count, probs=probs)
    scipy_dist = torch_dist.scipy_nbinom

    torch_icdf = torch_dist.icdf(
        torch.as_tensor(value, dtype=torch.float64)
    ).numpy()
    scipy_icdf = scipy_dist.ppf(np.asarray(value))

    assert np.allclose(torch_icdf, scipy_icdf)
