from models.base_gls import ControlInputs
import os
import inspect
from copy import copy
from typing import List, NamedTuple, Union
from shutil import copyfile
from collections import namedtuple
import math
from box import Box
import datetime
import numpy as np
import torch
import torch.distributions
from torch import nn
from torch.distributions import Independent, Normal, Bernoulli
from torch.nn.init import orthogonal_

from torch_extensions.ops import batch_diag_matrix, batch_diag

LOG_2PI = math.log(2 * math.pi)


def make_dummy_ssm_params():
    a1, a2 = 0.5, -1.0

    # state-related parameters
    A = torch.tensor([[a1, a2], [1.0, 0.0]])
    B = torch.tensor([[1.0], [1.0]])
    R = 0.064 * torch.eye(2)

    # observation/emission-related parameters
    C = torch.tensor([[2.0, 1.0]])
    D = torch.tensor([[1.0, 1.0, 1.0]])
    Q = 0.1 * torch.eye(1)

    # initial state
    m0 = torch.zeros((2,))
    V0 = torch.eye(2) * 4.0

    params = Box(
        {"A": A, "B": B, "C": C, "D": D, "R": R, "Q": Q, "m0": m0, "V0": V0,}
    )
    return params


def make_dummy_input_data(ssm_params, n_data=100, n_timesteps=10):
    device, dtype = ssm_params.A.device, ssm_params.A.dtype
    n_ctrlx = ssm_params.B.shape[1]
    n_ctrly = ssm_params.D.shape[1]
    x = 0.01 * torch.randn(
        *(n_timesteps, n_data, n_ctrlx), dtype=dtype, device=device,
    )
    y = 0.01 * torch.randn(
        *(n_timesteps, n_data, n_ctrly), dtype=dtype, device=device,
    )
    input_data = ControlInputs(state=x, target=y)
    return input_data


def make_inv_tril_parametrization(mat):
    L = torch.cholesky(mat)
    Linv = torch.inverse(L)
    Linv_tril = torch.tril(Linv, -1)
    Linv_logdiag = torch.log(batch_diag(Linv))
    return Linv_tril, Linv_logdiag


def make_inv_tril_parametrization_from_cholesky(Lmat):
    Linv = torch.inverse(Lmat)
    Linv_tril = torch.tril(Linv, -1)
    Linv_logdiag = torch.log(batch_diag(Linv))
    return Linv_tril, Linv_logdiag


def convert_ssm_params_to_model_params(A, B, C, D, R, Q, m0, V0):
    LRinv_tril, LRinv_logdiag = make_inv_tril_parametrization(R)
    LQinv_tril, LQinv_logdiag = make_inv_tril_parametrization(Q)
    LV0inv_tril, LV0inv_logdiag = make_inv_tril_parametrization(V0)

    params = Box(
        A=A,
        B=B,
        C=C,
        D=D,
        m0=m0,
        LV0inv_tril=LV0inv_tril,
        LV0inv_logdiag=LV0inv_logdiag,
        LRinv_tril=LRinv_tril,
        LRinv_logdiag=LRinv_logdiag,
        LQinv_tril=LQinv_tril,
        LQinv_logdiag=LQinv_logdiag,
    )
    model_params = Box(
        {
            name: torch.nn.Parameter(val.clone().detach())
            for name, val in params.items()
        }
    )

    return model_params


def convert_model_params_to_ssm_params(
    A,
    B,
    C,
    D,
    m0,
    LV0inv_tril,
    LV0inv_logdiag,
    LRinv_tril,
    LRinv_logdiag,
    LQinv_tril,
    LQinv_logdiag,
):
    R = torch.cholesky_inverse(
        torch.tril(LRinv_tril, -1) + torch.diag(torch.exp(LRinv_logdiag))
    )
    Q = torch.cholesky_inverse(
        torch.tril(LQinv_tril, -1) + torch.diag(torch.exp(LQinv_logdiag))
    )
    V0 = torch.cholesky_inverse(
        torch.tril(LV0inv_tril, -1) + torch.diag(torch.exp(LV0inv_logdiag))
    )
    return Box(A=A, B=B, C=C, D=D, R=R, Q=Q, m0=m0, V0=V0)


def add_sample_dims_to_initial_state(m0, V0, dims):
    if not m0.ndim == V0.ndim - 1:
        raise Exception(
            f"m0 should be (batched) vector, V0 (batched) var, but"
            f"dims don't match. Got m0: {m0.ndim}, V0: {V0.ndim}."
        )
    dim_particle = (
        (dims.particle,)
        if dims.particle is not None and dims.particle != 0
        else tuple()
    )
    if m0.ndim == 1:  # add batch dim (trade off performance for consistency)
        m0 = m0.repeat(dim_particle + (dims.batch,) + (1,) * m0.ndim)
        V0 = V0.repeat(dim_particle + (dims.batch,) + (1,) * V0.ndim)
    return m0, V0


class TensorDims(NamedTuple):
    timesteps: int
    particle: int
    batch: int
    state: int
    target: int
    ctrl_target: Union[int, None]
    ctrl_state: Union[int, None]
    ctrl_switch: Union[int, None] = None
    switch: Union[int, None] = None
    auxiliary: Union[int, None] = None
    anomaly: Union[int, None] = None
    timefeat: Union[int, None] = None
    staticfeat: Union[int, None] = None
    cat_embedding: Union[int, None] = None


class Lambda(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


class SigmoidLimiter(nn.Module):
    def __init__(self, limits):
        super().__init__()
        assert len(limits) == 2 and limits[0] < limits[1]
        self.offset = limits[0]
        self.scaling = limits[1] - limits[0]

    def forward(self, x):
        # factor 4 to make it tanh with derivative 1 at origin.
        return self.offset + self.scaling * torch.sigmoid(
            x * (4 / self.scaling)
        )


def compute_conv_output_img_dims(
    dims_img: (list, tuple),
    kernel_size: (int, tuple),
    stride: (int, tuple),
    padding: (int, tuple),
    dilation: (int, tuple, None) = None,
) -> tuple:
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    elif dilation is None:
        dilation = (
            1,
            1,
        )

    dims_img_in = list(dims_img)
    # dims_img_in = [shp_in[1], shp_in[2]]
    dims_img_out = copy(dims_img_in)

    for idx in range(len(dims_img_out)):
        dims_img_out[idx] = (
            dims_img_out[idx]
            + 2 * padding[idx]
            - dilation[idx] * (kernel_size[idx] - 1)
            - 1
        ) // stride[idx] + 1
    return tuple(dims_img_out)


def compute_cnn_output_filters_and_dims(
    dims_img, dims_filter, kernel_sizes, strides, paddings, dilations=None,
):
    dims_img = dims_img[-2:]
    for idx_hidden in range(len(dims_filter)):
        dims_img = compute_conv_output_img_dims(
            dims_img=dims_img,
            kernel_size=kernel_sizes[idx_hidden],
            stride=strides[idx_hidden],
            padding=paddings[idx_hidden],
            dilation=dilations[idx_hidden] if dilations is not None else None,
        )
    shp_enc_out = (dims_filter[-1],) + dims_img
    dim_out_flat_filter = int(np.prod(shp_enc_out))
    return shp_enc_out, dim_out_flat_filter


class Reshape(nn.Module):
    def __init__(self, shape, n_batch_dims=1):
        super().__init__()
        self.shape = shape  # does not include batch dims
        self.n_batch_dims = n_batch_dims

    def forward(self, x):
        return x.reshape(x.shape[: self.n_batch_dims] + self.shape)


class BatchDiagMatrix(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, diagonal):
        return batch_diag_matrix(diagonal=diagonal)


class Bias(nn.Module):
    def __init__(self, dim_out, init_val, requires_grad=True):
        super().__init__()
        self._value = nn.Parameter(
            init_val * torch.ones(dim_out), requires_grad=requires_grad
        )

    def forward(self, x):
        assert x.ndim == 2, "not implemented for CNN"
        return self._value.repeat(x.shape[:-1] + (1,))


class IndependentNormal(Independent):
    """
    Diagonal Normal that can be used similar to Multivariate Normal but is more
    memory and computationally efficient. Cholesky -> sqrt. Inverse -> **-1.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(Normal(*args, **kwargs), 1)

    @property
    def loc(self):
        return self.base_dist.loc

    @property
    def covariance_matrix(self):
        return batch_diag_matrix(self.base_dist.variance)

    @property
    def scale_tril(self):
        return batch_diag_matrix(self.base_dist.scale)

    @property
    def precision_matrix(self):
        return batch_diag_matrix(self.base_dist.variance ** -1)


class IndependentBernoulli(Independent):
    def __init__(self, *args, **kwargs):
        super().__init__(Bernoulli(*args, **kwargs), 1)


def create_zeros_state_vec(dims, device, dtype):
    dim_particle = (
        (dims.particle,)
        if dims.particle is not None and dims.particle != 0
        else tuple()
    )
    vec = [
        torch.zeros(
            dim_particle + (dims.batch, dims.state,),
            device=device,
            dtype=dtype,
        )
        for t in range(dims.timesteps)
    ]
    return vec


def create_zeros_state_mat(dims, device, dtype):
    dim_particle = (
        (dims.particle,)
        if dims.particle is not None and dims.particle != 0
        else tuple()
    )
    mat = [
        torch.zeros(
            dim_particle + (dims.batch, dims.state, dims.state),
            device=device,
            dtype=dtype,
        )
        for t in range(dims.timesteps)
    ]
    return mat


def create_zeros_switch_vec(dims, device, dtype):
    dim_particle = (
        (dims.particle,)
        if dims.particle is not None and dims.particle != 0
        else tuple()
    )
    vec = [
        torch.zeros(
            dim_particle + (dims.batch, dims.switch,),
            device=device,
            dtype=dtype,
        )
        for t in range(dims.timesteps)
    ]
    return vec


def create_zeros_log_weights(dims, device, dtype):
    dim_particle = (
        (dims.particle,)
        if dims.particle is not None and dims.particle != 0
        else tuple()
    )
    log_weights = [
        torch.zeros(dim_particle + (dims.batch,), device=device, dtype=dtype)
        for t in range(dims.timesteps)
    ]
    return log_weights


class ConditionalTransformation(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, n_base_transforms: int):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.n_base_transforms = n_base_transforms


class ConditionalAffine(ConditionalTransformation):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        n_base_transforms: int,
        init_fn: callable = orthogonal_,
        bias: bool = True,
    ):
        super().__init__(
            dim_in=dim_in, dim_out=dim_out, n_base_transforms=n_base_transforms
        )

        mats_init = [
            init_fn(torch.empty(dim_out, dim_in))
            for _ in range(n_base_transforms)
        ]
        self.base_matrices = nn.Parameter(torch.stack(mats_init, dim=0))
        if bias:
            self.base_biases = nn.Parameter(
                torch.zeros(n_base_transforms, dim_out)
            )
        else:
            self.register_parameter("base_biases", None)

    def forward(
        self, inputs: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        linear = torch.einsum(
            "...k,...i,koi->...o", weights, inputs, self.base_matrices
        )
        if self.base_biases is not None:
            bias = torch.einsum("...k,ko->...o", weights, self.base_biases)
            affine = linear + bias
        else:
            affine = linear
        return affine


def flatten_iterable(items: (list, tuple, set)):
    flattened_items = []
    for item in items:
        if isinstance(item, (list, tuple, set)):
            flattened_items.extend(flatten_iterable(item))
        else:
            flattened_items.append(item)
    return tuple(flattened_items)


def one_hot(labels, num_classes, is_squeeze=False):
    if not is_squeeze:
        return one_hot(
            labels.squeeze(dim=-1), num_classes=num_classes, is_squeeze=True
        )
    else:
        return torch.eye(num_classes, dtype=labels.dtype)[labels]


def prepare_logging(
    consts,
    config,
    run_nr: (int, None) = None,
    copy_config_file=True,
    root_log_path=None,
):
    if root_log_path is None:  # create
        if run_nr is None:  # time-stamped folders inside folder "other"
            now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            cfg_save_path = os.path.join(
                consts.log_dir, config.dataset_name, "other", now
            )
            root_log_path = cfg_save_path
        else:  # dataset_name/experiment_name/run_nr/.
            cfg_save_path = os.path.join(
                consts.log_dir, config.dataset_name, config.experiment_name
            )
            root_log_path = os.path.join(cfg_save_path, str(run_nr))

        plot_path = os.path.join(root_log_path, "plots")
        model_snapshots_path = os.path.join(
            root_log_path, "parameters_over_time"
        )
        metrics_path = os.path.join(root_log_path, "metrics")

        os.makedirs(root_log_path, exist_ok=False)
        os.makedirs(plot_path, exist_ok=False)
        os.makedirs(model_snapshots_path, exist_ok=False)
        os.makedirs(metrics_path, exist_ok=False)

        if copy_config_file:
            copyfile(
                src=os.path.abspath(inspect.getfile(config.__class__)),
                dst=os.path.join(cfg_save_path, "config.py"),
            )
            copyfile(
                src=os.path.abspath(consts.__file__),
                dst=os.path.join(cfg_save_path, "consts.py"),
            )

    else:  # load
        plot_path = os.path.join(root_log_path, "plots")
        model_snapshots_path = os.path.join(
            root_log_path, "parameters_over_time"
        )
        metrics_path = os.path.join(root_log_path, "metrics")

    log_paths = namedtuple("log_paths", ["root", "plot", "model", "metrics"])
    return log_paths(
        root=root_log_path,
        plot=plot_path,
        model=model_snapshots_path,
        metrics=metrics_path,
    )


def list_of_dicts_to_dict_of_list(list_of_dicts: List[dict]):
    assert isinstance(list_of_dicts, (list, tuple))
    assert isinstance(list_of_dicts[0], dict)
    assert all(
        list_of_dicts[0].keys() == list_of_dicts[idx].keys()
        for idx in range(len(list_of_dicts))
    )
    dict_of_list = {
        key: [list_of_dicts[idx][key] for idx in range(len(list_of_dicts))]
        for key in list_of_dicts[0].keys()
    }
    return dict_of_list


def shorten_iter(it, steps: int):
    for idx, val in enumerate(it):
        if idx >= steps:
            break
        else:
            yield val
