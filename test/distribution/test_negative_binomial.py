import mxnet as mx
import numpy as np
from gluonts.distribution.neg_binomial import NegativeBinomialOutput


def test_issue_287():
    network_output = mx.nd.ones(shape=(10,))
    distr_output = NegativeBinomialOutput()
    args_proj = distr_output.get_args_proj()
    args_proj.initialize(init=mx.init.Constant(-1e2))
    distr_args = args_proj(network_output)
    distr = distr_output.distribution(distr_args)
    x = mx.nd.array([1.0])
    ll = distr.log_prob(x)
    assert np.isfinite(ll.asnumpy())
