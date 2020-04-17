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

# Standard library imports
import inspect
import os
import signal
import tempfile
import time
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    cast,
    Union,
    Tuple,
    Iterable,
)

# Third-party imports
import mxnet as mx
import numpy as np
from mxnet.gluon.block import _flatten

# First-party imports
from gluonts.core.serde import dump_json, load_json
from gluonts.model.common import Tensor

MXNET_HAS_ERF = hasattr(mx.nd, "erf")
MXNET_HAS_ERFINV = hasattr(mx.nd, "erfinv")


class Timer:
    """Context manager for measuring the time of enclosed code fragments."""

    def __enter__(self):
        self.start = time.perf_counter()
        self.interval = None
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.interval = self.end - self.start


class SignalHandler:
    """
    A context manager that attaches a set of signal handlers within its scope.

    Parameters
    ----------
    handlers_map
        A dictionary mapping signal numbers to associated signal handlers to
        be attached within the scope of the enclosing `SignalHandler` instance.
    """

    Callback = Optional[Callable[[int, Any], None]]

    def __init__(self, handlers_map: Dict[int, Callback]) -> None:
        self.handlers_map = handlers_map

    def __enter__(self):
        self.default_handlers = {
            s: signal.signal(s, h) for s, h in self.handlers_map.items()
        }
        return self

    def __exit__(self, *args):
        for s, h in self.default_handlers.items():
            signal.signal(s, h)


class HybridContext:
    """
    A context manager that ensures that an MXNet network is operating in a
    hybridized / not hybridized mode.

    Parameters
    ----------
    net
        The network whose hybrid mode has to be modified within the enclosing
        context.
    hybridize
        A boolean flag inidicating whether the hybrid mode should be set or
        not.
    kwargs
        A dictionary of optional arguments to pass to the `hybridize()` call
        of the enclosed `HybridBlock` network.
    """

    def __init__(
        self,
        net: mx.gluon.HybridBlock,
        hybridize: bool,
        data_batch: Optional[List[mx.nd.NDArray]] = None,
        **kwargs,
    ) -> None:
        self.net = net
        self.required_mode = hybridize

        self.original_mode = getattr(net, "_active", False)
        self.data_batch = data_batch
        self.kwargs = kwargs

    def __enter__(self):
        self.net.hybridize(active=self.required_mode, **self.kwargs)
        if self.data_batch is not None:
            self.net(*self.data_batch)

    def __exit__(self, *args):
        self.net.hybridize(active=self.original_mode, **self.kwargs)


def maybe_len(obj) -> Optional[int]:
    try:
        return len(obj)
    except NotImplementedError:
        return None


def copy_parameters(
    net_source: mx.gluon.Block,
    net_dest: mx.gluon.Block,
    ignore_extra: bool = False,
    allow_missing: bool = False,
) -> None:
    """
    Copies parameters from one network to another.

    Parameters
    ----------
    net_source
        Input network.
    net_dest
        Output network.
    ignore_extra
        Whether to ignore parameters from the source that are not
        present in the target.
    allow_missing
        Whether to allow additional parameters in the target not
        present in the source.
    """
    with tempfile.TemporaryDirectory(
        prefix="gluonts-estimator-temp-"
    ) as model_dir:
        model_dir_path = str(Path(model_dir) / "tmp_model")
        net_source.save_parameters(model_dir_path)
        net_dest.load_parameters(
            model_dir_path,
            ctx=mx.current_context(),
            allow_missing=allow_missing,
            ignore_extra=ignore_extra,
        )


def get_hybrid_forward_input_names(hb: mx.gluon.HybridBlock):
    params = inspect.signature(hb.hybrid_forward).parameters
    param_names = list(params)
    assert param_names[0] == "F", (
        f"Expected first argument of HybridBlock to be `F`, "
        f"but found `{param_names[0]}`"
    )
    return param_names[1:]  # skip: F


# noinspection PyProtectedMember
def hybrid_block_to_symbol_block(
    hb: mx.gluon.HybridBlock, data_batch: List[mx.nd.NDArray]
) -> mx.gluon.SymbolBlock:
    """
    Converts a Gluon `HybridBlock` to a `SymbolBlock`. Following the Gluon API,
    this is achieved by a `hybridize()` call on the passed `HybridBlock`, a
    single forward pass (using the provided data batch), and a combination of
    an `export()` and an `import()` calls of the input block.

    Note that MXNet has `problems with this method
    <https://github.com/apache/incubator-mxnet/issues/12783>`_.

    Parameters
    ----------
    hb
        The Gluon `HybridBlock` to convert.
    data_batch
        Data to use for the forward pass after the `hybridize()` call.

    Returns
    -------
    mx.gluon.SymbolBlock
        The resulting Gluon block backed by an MXNet symbol graph.
    """
    with tempfile.TemporaryDirectory(
        prefix="gluonts-estimator-temp-"
    ) as model_dir:
        # when importing, SymbolBlock has to know about the total number
        # of input symbols, including nested Tensors
        flat_data_batch, _ = _flatten(data_batch, "input")
        num_inputs = len(flat_data_batch)

        model_dir_path = Path(model_dir)
        model_name = "gluonts-model"

        with HybridContext(
            net=hb,
            hybridize=True,
            data_batch=data_batch,
            static_alloc=True,
            static_shape=True,
        ):
            export_symb_block(hb, model_dir_path, model_name)
            sb = import_symb_block(num_inputs, model_dir_path, model_name)

        return sb


# noinspection PyProtectedMember
def export_symb_block(
    hb: mx.gluon.HybridBlock, model_dir: Path, model_name: str, epoch: int = 0
) -> None:
    """
    Serializes a hybridized Gluon `HybridBlock`.

    Parameters
    ----------
    hb
        The block to export.
    model_dir
        The path where the model will be saved.
    model_name
        The name identifying the model.
    epoch
        The epoch number, which together with the `model_name` identifies the
        model parameters.
    """
    hb.export(path=str(model_dir / model_name), epoch=epoch)

    # FIXME: we persist input/output formats of hybrid blocks as mxnet does not
    # FIXME: https://github.com/apache/incubator-mxnet/issues/17488
    with (model_dir / f"{model_name}-in_out_format.json").open("w") as fp:
        in_out_format = dict(
            in_format=hb._in_format, out_format=hb._out_format
        )
        print(dump_json(in_out_format), file=fp)


def import_symb_block(
    num_inputs: int, model_dir: Path, model_name: str, epoch: int = 0
) -> mx.gluon.SymbolBlock:
    """
    Deserializes a hybridized Gluon `HybridBlock` as a `SymbolBlock`.

    Parameters
    ----------
    num_inputs
        The number of inputs of the serialized block.
    model_dir
        The path where the model is saved.
    model_name
        The name identifying the model.
    epoch
        The epoch number, which together with the `model_name` identifies the
        model parameters.

    Returns
    -------
    mx.gluon.SymbolBlock
        The deserialized block.
    """
    if num_inputs == 1:
        input_names = ["data"]
    else:
        input_names = [f"data{i}" for i in range(num_inputs)]

    # FIXME: prevents mxnet from failing with empty saved parameters list
    # FIXME: https://github.com/apache/incubator-mxnet/issues/17488
    param_file: Optional[str] = str(
        model_dir / f"{model_name}-{epoch:04}.params"
    )
    if not mx.nd.load(param_file):
        param_file = None

    # FIXME: mx.gluon.SymbolBlock cannot infer float_type and uses default np.float32
    # FIXME: https://github.com/apache/incubator-mxnet/issues/11849
    sb = mx.gluon.SymbolBlock.imports(
        symbol_file=str(model_dir / f"{model_name}-symbol.json"),
        input_names=input_names,
        param_file=param_file,
        ctx=mx.current_context(),
    )

    # FIXME: try to retrieve input/output format
    # FIXME: https://github.com/apache/incubator-mxnet/issues/17488
    format_json_path = model_dir / f"{model_name}-in_out_format.json"
    if format_json_path.exists():
        with format_json_path.open("r") as fp:
            formats = load_json(fp.read())
            sb._in_format = formats["in_format"]
            sb._out_format = formats["out_format"]

    return sb


def export_repr_block(
    rb: mx.gluon.HybridBlock, model_dir: Path, model_name: str, epoch: int = 0
) -> None:
    """
    Serializes a representable Gluon block.

    Parameters
    ----------
    rb
        The block to export.
    model_dir
        The path where the model will be saved.
    model_name
        The name identifying the model.
    epoch
        The epoch number, which together with the `model_name` identifies the
        model parameters.
    """
    with (model_dir / f"{model_name}-network.json").open("w") as fp:
        print(dump_json(rb), file=fp)
    rb.save_parameters(str(model_dir / f"{model_name}-{epoch:04}.params"))


def import_repr_block(
    model_dir: Path, model_name: str, epoch: int = 0
) -> mx.gluon.HybridBlock:
    """
    Deserializes a representable Gluon block.

    Parameters
    ----------
    model_dir
        The path where the model is saved.
    model_name
        The name identifying the model.
    epoch
        The epoch number, which together with the `model_name` identifies the
        model parameters.

    Returns
    -------
    mx.gluon.HybridBlock:
        The deserialized block.
    """
    with (model_dir / f"{model_name}-network.json").open("r") as fp:
        rb = cast(mx.gluon.HybridBlock, load_json(fp.read()))
    rb.load_parameters(
        str(model_dir / f"{model_name}-{epoch:04}.params"),
        ctx=mx.current_context(),
        allow_missing=False,
        ignore_extra=False,
    )
    return rb


def cumsum(
    F, x: Tensor, exclusive: bool = False, reverse: bool = False
) -> Tensor:
    r"""
    Find cumulative sum on the last axis by multiplying with lower triangular
    ones-matrix:

    .. math::

       \operatorname{cumsum}(x) =
       \begin{cases}
         \operatorname{ltr\_ones} \times x
           & \text{for cumulative sum}\\
         x \times \operatorname{ltr\_ones}
           & \text{for cumulative sum in the reverse order}
       \end{cases}

    Also supports `exclusive` flag to start the cumsum with zero.
    For example, if :math:`x = [a, b, c]`, we have

    .. math::

       \operatorname{cumsum}(x) =
       \begin{cases}
         [a, a + b, a + b + c]
           & \text{if }\mathit{reverse = False, exclusive = False}\\
         [0, a, a + b]
           & \text{if }\mathit{reverse = False, exclusive = True}\\
         [a + b + c, b + c, c]
           & \text{if }\mathit{reverse = True, exclusive = False}\\
         [b + c, c, 0]
           & \text{if }\mathit{reverse = True, exclusive = True}\\
       \end{cases}

    Parameters
    ----------
    F
        The function space to use.
    x
        A tensor with shape :math:`(..., n)`.
    exclusive
        If `True`, the cumulative sum starts with zero.
    reverse
        If `True`, the cumulative sum is performed in the opposite direction.

    Returns
    -------
    Tensor:
        A modified tensor with identical shape and cumulative sums in the last
        axis.
    """

    # Create a new axis (for matrix multiplication) either at last location or
    # last-but-one location (for reverse mode)
    exp_dim = -2 if reverse else -1
    # (..., 1, n) if reverse is True and (..., n, 1) otherwise
    x = x.expand_dims(axis=exp_dim)

    # Ones_matrix (..., n, n)
    ones_matrix = F.linalg_gemm2(
        F.ones_like(x),
        F.ones_like(x),
        transpose_a=reverse,
        transpose_b=not reverse,
    )
    cumulative_sum = F.linalg_trmm(ones_matrix, x, rightside=reverse)

    if exclusive:
        cumulative_sum = cumulative_sum - x

    return cumulative_sum.squeeze(axis=exp_dim)


def weighted_average(
    F, x: Tensor, weights: Optional[Tensor] = None, axis: Optional[int] = None
) -> Tensor:
    """
    Computes the weighted average of a given tensor across a given axis, masking values associated with weight zero,
    meaning instead of `nan * 0 = nan` you will get `0 * 0 = 0`.

    Parameters
    ----------
    F
        The function space to use.
    x
        Input tensor, of which the average must be computed.
    weights
        Weights tensor, of the same shape as `x`.
    axis
        The axis along which to average `x`

    Returns
    -------
    Tensor:
        The tensor with values averaged along the specified `axis`.
    """
    if weights is not None:
        weighted_tensor = F.where(
            condition=weights, x=x * weights, y=F.zeros_like(x)
        )
        sum_weights = F.maximum(1.0, weights.sum(axis=axis))
        return weighted_tensor.sum(axis=axis) / sum_weights
    else:
        return x.mean(axis=axis)


def make_nd_diag(F, x: Tensor, d: int) -> Tensor:
    """
    Make a diagonal tensor, given the diagonal

    Parameters
    ----------
    F
        The function space to use.
    x
        Diagonal to use, shape :math:`(..., d)`.
    d
        Last dimension of `x`.

    Returns
    -------
    Tensor
        A tensor y of shape :math:`(..., d, d)` such that
        :math:`y[..., i, i] = x[..., i]`.
    """
    return F.broadcast_mul(F.eye(d), x.expand_dims(axis=-1))


def _broadcast_param(param, axes, sizes):
    for axis, size in zip(axes, sizes):
        param = param.expand_dims(axis=axis).broadcast_axes(
            axis=axis, size=size
        )

    return param


def erf(F, x: Union[Tensor, np.array]) -> Union[Tensor, np.array]:
    if F is mx.nd or F is mx.sym:
        if MXNET_HAS_ERF:
            return F.erf(x)
    # Using numerical recipes approximation for erf function
    # accurate to 1E-7

    ones = F.ones_like(x)
    zeros = F.zeros_like(x)

    t = ones / (ones + 0.5 * F.abs(x))

    coefficients = [
        1.00002368,
        0.37409196,
        0.09678418,
        -0.18628806,
        0.27886807,
        -1.13520398,
        1.48851587,
        -0.82215223,
        0.17087277,
    ]

    inner = zeros
    for c in coefficients[::-1]:
        inner = t * (c + inner)

    res = ones - t * F.exp((inner - 1.26551223 - F.square(x)))
    return F.where(x >= zeros, res, -1.0 * res)


def erfinv(F, x: Union[Tensor, np.array]) -> Union[Tensor, np.array]:
    if F is mx.nd or F is mx.sym:
        if MXNET_HAS_ERFINV:
            return F.erfinv(x)

    zeros = F.zeros_like(x)

    w = -F.log((1.0 - x) * (1.0 + x))
    mask_lesser = w < (zeros + 5.0)

    w = F.where(mask_lesser, w - 2.5, F.sqrt(w) - 3.0)

    coefficients_lesser = [
        2.81022636e-08,
        3.43273939e-07,
        -3.5233877e-06,
        -4.39150654e-06,
        0.00021858087,
        -0.00125372503,
        -0.00417768164,
        0.246640727,
        1.50140941,
    ]

    coefficients_greater_equal = [
        -0.000200214257,
        0.000100950558,
        0.00134934322,
        -0.00367342844,
        0.00573950773,
        -0.0076224613,
        0.00943887047,
        1.00167406,
        2.83297682,
    ]

    p = F.where(
        mask_lesser,
        coefficients_lesser[0] + zeros,
        coefficients_greater_equal[0] + zeros,
    )

    for c_l, c_ge in zip(
        coefficients_lesser[1:], coefficients_greater_equal[1:]
    ):
        c = F.where(mask_lesser, c_l + zeros, c_ge + zeros)
        p = c + p * w

    return p * x


def get_download_path() -> Path:
    """

    Returns
    -------
    Path
        default path to download datasets or models of gluon-ts.
        The path is either $MXNET_HOME if the environment variable is defined or
        /home/username/.mxnet/gluon-ts/
    """
    return Path(
        os.environ.get("MXNET_HOME", str(Path.home() / ".mxnet" / "gluon-ts"))
    )


def map_dct_values(fn: Callable, dct: dict) -> dict:
    """Maps `fn` over a dicts values."""
    return {key: fn(value) for key, value in dct.items()}


def assert_shape(x: Tensor, expected_shape: Tuple[int, ...]):
    """
    Assert expected shape if mode is mx.nd.

    Parameters
    ----------
    x
        Input Tensor
    expected_shape
        Expected shape
    Returns
    -------

    """
    if isinstance(x, mx.nd.NDArray):
        for i, j in zip(x.shape, expected_shape):
            if j != -1:
                assert (
                    i == j
                ), f"shape mismatch got {x.shape} expected {expected_shape}"
