from typing import Sequence, Optional, Union
from dataclasses import dataclass
from abc import ABCMeta

import torch
from torch import nn


@dataclass
class ControlInputs:
    state: torch.Tensor
    target: torch.Tensor

    def __getitem__(self, item):
        return self.__class__(**{k: v[item] for k, v in self.__dict__.items()})

    def __len__(self):
        if not len(set(len(v) for v in self.__dict__.values())) == 1:
            raise Exception("Not all data in this class has same length.")
        return len(self.state)

    def to(self, device=None, dtype=None):
        if device is None:
            device = self.state.device
        if dtype is None:
            dtype = self.state.dtype

        for key in self.__dict__.keys():
            val = getattr(self, key)
            # None (optional) or integer types are ignored.
            if isinstance(val, (torch.FloatTensor, torch.DoubleTensor)):
                setattr(self, key, val.to(dtype).to(device))
        return self


@dataclass
class GLSVariables:
    """ Stores either (m, V) or samples or both from a MultivariateNormal. """
    # Note: The performance difference to using a Multivariate
    # (computes choleksy and broadcasts) is actually small.
    # Could replace (m, V) by MultivariateNormal.

    # Setting default value not possible since subclasses of this dataclass
    # would need to set all fields then with default values too.
    m: (torch.Tensor, None)
    V: (torch.Tensor, None)
    x: (torch.Tensor, None)

    def __post_init__(self):
        has_state_dist_params = tuple(
            param is not None for param in (self.m, self.V)
        )
        if not len(set(has_state_dist_params)) == 1:
            raise Exception("Provide either all or no distribution parameters")

        has_state_sample = self.x is not None
        if not (all(has_state_dist_params) or has_state_sample):
            raise Exception("Provide at least either dist params or samples.")


@dataclass
class Latents:
    variables: GLSVariables


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
        n_target,
        n_ctrl_state=None,
        n_ctrl_target=None,
        n_particle=None,
    ):
        super().__init__()
        self.n_state = n_state
        self.n_target = n_target
        self.n_ctrl_state = n_ctrl_state
        self.n_ctrl_target = n_ctrl_target
        self.n_particle = n_particle

    def filter(
        self,
        past_targets: [Sequence[torch.Tensor], torch.Tensor],
        past_controls: Optional[Union[Sequence[ControlInputs], ControlInputs]] = None,
    ) -> Sequence[Latents]:
        raise NotImplementedError("Should be implemented by child class")

    def smooth(
        self,
        past_targets: [Sequence[torch.Tensor], torch.Tensor],
        past_controls: Optional[Union[Sequence[ControlInputs], ControlInputs]] = None,
    ) -> Sequence[Latents]:
        raise NotImplementedError("Should be implemented by child class")

    def sample(
        self,
        n_steps_forecast: int,
        n_batch: int,
        future_controls: Optional[Union[Sequence[torch.Tensor], ControlInputs]],
        deterministic: bool,
        **kwargs,
    ) -> Sequence[Prediction]:
        raise NotImplementedError("Should be implemented by child class")

    def forecast(
        self,
        n_steps_forecast: int,
        initial_latent: Latents,
        future_controls: Optional[Union[Sequence[ControlInputs], ControlInputs]],
        deterministic: bool,
    ) -> Sequence[Prediction]:
        raise NotImplementedError("Should be implemented by child class")

    def predict(
        self,
        n_steps_forecast: int,
        past_targets: [Sequence[torch.Tensor], torch.Tensor],
        past_controls: Optional[Union[Sequence[ControlInputs], ControlInputs]],
        future_controls: Optional[Union[Sequence[ControlInputs], ControlInputs]],
        deterministic: bool,
    ) -> Sequence[Prediction]:
        raise NotImplementedError("Should be implemented by child class")

    def loss(
        self,
        past_targets: [Sequence[torch.Tensor], torch.Tensor],
        past_controls: Optional[Union[Sequence[ControlInputs], ControlInputs]] = None,
    ) -> torch.Tensor:
        raise NotImplementedError("Should be implemented by child class")

