from typing import Tuple
from enum import Enum
from utils.utils import TensorDims
from dataclasses import dataclass


class SwitchLinkType(Enum):
    identity = 1
    shared = 2
    individual = 3


class TimeFeatType(Enum):
    timefeat = 1
    seasonal_indicator = 2
    both = 3
    none = 4


@dataclass()
class BaseConfig:
    """ Contains possible configs that all experiments share and should set """

    experiment_name: str
    dataset_name: str
    dims: TensorDims
    init_scale_A: (float, None)
    init_scale_B: (float, None)
    init_scale_C: (float, None)
    init_scale_D: (float, None)
    init_scale_R_diag: (float, Tuple[int, int])
    init_scale_Q_diag: (float, Tuple[int, int])
    init_scale_S_diag: (float, Tuple[int, int], None)
    state_prior_scale: float
    state_prior_loc: float
    switch_link_type: SwitchLinkType
    switch_link_dims_hidden: tuple
    switch_link_activations: tuple
    n_base_A: (int, None)
    n_base_B: (int, None)
    n_base_C: (int, None)
    n_base_D: (int, None)
    n_base_R: (int, None)
    n_base_Q: (int, None)
    n_base_F: (int, None)
    n_base_S: (int, None)
    requires_grad_R: bool
    requires_grad_Q: bool
    n_epochs: int
    lr: float
    LRinv_logdiag_scaling: float
    LQinv_logdiag_scaling: float
    B_scaling: float
    D_scaling: float
    eye_init_A: bool
