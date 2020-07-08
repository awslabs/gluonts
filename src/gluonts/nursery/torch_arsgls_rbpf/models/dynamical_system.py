from abc import ABCMeta
from torch import nn
from utils.utils import TensorDims


class DynamicalSystem(nn.Module, metaclass=ABCMeta):
    def __init__(
        self,
        n_state,
        n_obs,
        n_ctrl_state=None,
        n_ctrl_obs=None,
        n_particle=None,
    ):
        super().__init__()
        self.n_state = n_state
        self.n_target = n_obs
        self.n_ctrl_state = n_ctrl_state
        self.n_ctrl_target = n_ctrl_obs
        self.n_particle = n_particle

    def get_dims(
        self, y=None, u_state=None, u_obs=None, n_timesteps=None, n_batch=None
    ):
        if y is not None:
            n_timesteps = y.shape[0]
            n_batch = y.shape[1]
        elif u_state is not None:
            n_timesteps = u_state.shape[0]
            n_batch = u_state.shape[1]
        elif u_obs is not None:
            n_timesteps = u_obs.shape[0]
            n_batch = u_obs.shape[1]
        else:
            if n_timesteps is None and n_batch is None:
                raise Exception(
                    "either provide n_timesteps and n_batch directly, "
                    "or provide any of (y, u_state, u_obs, u_switch). "
                    f"Got following types: "
                    f"y: {type(y)}, "
                    f"u_state: {type(u_state)}, "
                    f"u_obs: {type(u_obs)}, "
                    f"n_timesteps: {type(n_timesteps)}, "
                    f"n_batch: {type(n_batch)}"
                )
        return TensorDims(
            timesteps=n_timesteps,
            particle=self.n_particle,
            batch=n_batch,
            state=self.n_state,
            target=self.n_target,
            ctrl_target=self.n_ctrl_target,
            ctrl_state=self.n_ctrl_state,
        )

    def filter_forward(self, *args, **kwargs):
        raise NotImplementedError("Should be implemented by child class")

    def smooth_forward_backward(self, *args, **kwargs):
        raise NotImplementedError("Should be implemented by child class")

    def smooth_global(self, *args, **kwargs):
        raise NotImplementedError("Should be implemented by child class")

    def sample(self, *args, **kwargs):
        raise NotImplementedError("Should be implemented by child class")

    def loss_forward(self, *args, **kwargs):
        raise NotImplementedError("Should be implemented by child class")

    def loss_em(self, *args, **kwargs):
        raise NotImplementedError("Should be implemented by child class")
