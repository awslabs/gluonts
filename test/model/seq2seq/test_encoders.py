# Third-party imports
import pytest
from mxnet import nd

# First-party imports
from gluonts.block.encoder import HierarchicalCausalConv1DEncoder

nd_None = nd.array([])


@pytest.mark.skip()
def test_hierarchical_cnn_encoders() -> None:
    num_ts = 2
    ts_len = 10
    test_data = nd.arange(num_ts * ts_len).reshape(shape=(num_ts, ts_len, 1))

    chl_dim = [30, 30, 30]
    ks_seq = [3] * len(chl_dim)
    dial_seq = [1, 3, 9]

    cnn = HierarchicalCausalConv1DEncoder(
        dial_seq, ks_seq, chl_dim, use_residual=True
    )
    cnn.collect_params().initialize()
    cnn.hybridize()

    print(cnn(test_data, nd_None, nd_None)[1].shape)
