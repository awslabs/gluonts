from torch import nn
from torch_extensions.affine import Affine, Bias
from torch_extensions.batch_diag_matrix import BatchDiagMatrix
from torch_extensions.layers_with_init import Linear


class LinearWithShiftedInitScaleBias(Linear):
    def __init__(self, *args, weight_scaling=1.0, bias_offset=0.0, **kwargs):
        self._weight_scaling = weight_scaling
        self._bias_offset = bias_offset
        super().__init__(*args, **kwargs)

    def reset_parameters(self):
        super().reset_parameters()
        self.weight.data *= self._weight_scaling
        self.bias.data += self._bias_offset


class DefaultScaleTransform(nn.Module):
    def __init__(
            self,
            dim_in,
            dim_out,
            pre_rect_scaling=0.25,  # smaller variability of scale.
            pre_rect_bias=-2.0,  # hard-coded smaller std of std.
            make_diag_cov_matrix=True,  # Otherwise vector
    ):
        super().__init__()
        self.linear = LinearWithShiftedInitScaleBias(
            dim_in,
            dim_out,
            weight_scaling=pre_rect_scaling,
            bias_offset=pre_rect_bias,
        )
        self.rectifier = nn.Sequential(nn.Softplus(), Bias(loc=1e-6))
        self.diag_mat = BatchDiagMatrix() if make_diag_cov_matrix else None

    def forward(self, x):
        scale = self.rectifier(self.linear(x))
        return self.diag_mat(scale) if self.diag_mat is not None else scale
