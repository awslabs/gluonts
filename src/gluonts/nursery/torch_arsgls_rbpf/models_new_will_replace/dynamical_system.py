from dataclasses import dataclass
from abc import ABCMeta

import torch
from torch import nn

@dataclass
class Latents:
    pass


@dataclass
class Prediction:
    """
    Template for outputs of all our GLS-based models.
    Note that the latents and emission here have
    1 more dimension in all corresponding tensors, i.e. time-dimension.
    """

    latents: Latents
    emissions: (torch.Tensor, torch.distributions.Distribution)


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
        self.n_obs = n_obs
        self.n_ctrl_state = n_ctrl_state
        self.n_ctrl_obs = n_ctrl_obs
        self.n_particle = n_particle

    def filter(self, *args, **kwargs):
        raise NotImplementedError("Should be implemented by child class")

    def smooth(self, *args, **kwargs):
        raise NotImplementedError("Should be implemented by child class")

    def sample(self, *args, **kwargs):
        raise NotImplementedError("Should be implemented by child class")

    def forecast(self, *args, **kwargs):
        raise NotImplementedError("Should be implemented by child class")

    def predict(self, *args, **kwargs):
        raise NotImplementedError("Should be implemented by child class")

    def loss(self, *args, **kwargs):
        raise NotImplementedError("Should be implemented by child class")