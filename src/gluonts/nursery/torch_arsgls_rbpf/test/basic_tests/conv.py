import torch
from torch import nn

from utils.utils import compute_conv_output_img_dims


def test_compute_conv_dims_out():
    for width_img in [63, 64, 65, 66]:
        dims_img = (width_img, width_img)
        inp = torch.randn((10, 1,) + dims_img)
        for padding in [0, 1, 2]:
            for dilation in [1, 2, 3]:
                for stride in [1, 2, 3]:
                    for kernel_size in [2, 3, 4, 5]:
                        conv = nn.Conv2d(
                            in_channels=1,
                            out_channels=1,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding,
                            dilation=dilation,
                        )
                        computed_img_dims_out = compute_conv_output_img_dims(
                            dims_img=dims_img,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding,
                            dilation=dilation,
                        )
                        actual_img_dims_out = conv(inp).shape[2:]
                        assert computed_img_dims_out == actual_img_dims_out
