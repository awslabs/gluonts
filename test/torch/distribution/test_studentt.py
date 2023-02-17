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

import numpy as np
import torch

from gluonts.torch.distributions.studentT import StudentT


@pytest.mark.parametrize("df", [0.5, 1.5, 2.4])
@pytest.mark.parametrize("loc", [-3.0, 0.0])
@pytest.mark.parametrize("scale", [0.1, 0.8, 2.3])
def test_custom_studentt_logpdf_matches_scipy_student_logpdf(df, loc, scale):
    torch_dist = StudentT(df=df, loc=loc, scale=scale)
    scipy_dist = torch_dist.scipy_student_t
    x = torch.linspace(-20, 20, 1000)
    log_pdf_torch = torch_dist.log_prob(x).numpy()
    log_pdf_scipy = scipy_dist.logpdf(x.numpy())

    assert np.allclose(log_pdf_torch, log_pdf_scipy)


@pytest.mark.parametrize("df", [0.5, 1.5, 2.4])
@pytest.mark.parametrize("loc", [-3.0, 0.0])
@pytest.mark.parametrize("scale", [0.1, 0.8, 2.3])
@pytest.mark.parametrize("value", [0.1, 0.5, 0.9])
def test_custom_studentt_cdf(df, loc, scale, value):
    torch_dist = StudentT(df=df, loc=loc, scale=scale)
    scipy_dist = torch_dist.scipy_student_t

    torch_cdf = torch_dist.cdf(torch.as_tensor(value))
    scipy_cdf = scipy_dist.cdf(np.asarray(value))

    assert np.allclose(torch_cdf, scipy_cdf)


@pytest.mark.parametrize("df", [0.5, 1.5, 2.4])
@pytest.mark.parametrize("loc", [-3.0, 0.0])
@pytest.mark.parametrize("scale", [0.1, 0.8, 2.3])
@pytest.mark.parametrize("value", [0.1, 0.5, 0.9])
def test_custom_studentt_icdf(df, loc, scale, value):
    torch_dist = StudentT(df=df, loc=loc, scale=scale)
    scipy_dist = torch_dist.scipy_student_t

    torch_icdf = torch_dist.icdf(torch.as_tensor(value))
    scipy_icdf = scipy_dist.ppf(np.asarray(value))

    assert np.allclose(torch_icdf, scipy_icdf)
