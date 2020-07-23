from typing import Sequence, Optional, Union
from dataclasses import dataclass
from abc import ABCMeta
import torch
from torch import nn


@dataclass
class ControlInputs:
    state: torch.Tensor
    target: torch.Tensor

    def __len__(self):
        return len(self.state)

    def __getitem__(self, item):
        return self.__class__(**{k: v[item] if v is not None else None for k, v in self.__dict__.items()})

    def __iter__(self):
        for idx in range(len(self)):
            yield self.__class__(
                **{k: v[idx] if v is not None else None
                   for k, v in self.__dict__.items()},
            )

    def __post_init__(self):
        if not len(set(len(v) for v in self.__dict__.values() if v is not None)) == 1:
            raise Exception("Not all data in this class has same length.")

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
class GLSParams:  # TODO: better naming?
    A: torch.Tensor
    C: torch.Tensor
    Q: torch.Tensor
    R: torch.Tensor
    LQ: Optional[torch.Tensor] = None
    LR: Optional[torch.Tensor] = None
    B: Optional[torch.Tensor] = None
    D: Optional[torch.Tensor] = None
    b: Optional[torch.Tensor] = None
    d: Optional[torch.Tensor] = None

    def __len__(self):
        # assert all([
        #     len(self.A) == len(v)
        #     for v in self.__dict__.values() if v is not None
        # ])
        return len(self.A)

    def __getitem__(self, item):
        return self.__class__(**{k: v[item] if v is not None else None for k, v in self.__dict__.items()})

    def __iter__(self):
        for idx in range(len(self)):
            yield self.__class__(
                **{k: v[idx] if v is not None else None
                   for k, v in self.__dict__.items()},
            )

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
    Cov: (torch.Tensor, None)
    x: (torch.Tensor, None)

    def __len__(self):
        return len(self.m) if self.m is not None else len(self.x)

    def __getitem__(self, item):
        return self.__class__(**{k: v[item] if v is not None else None for k, v in self.__dict__.items()})

    def __iter__(self):
        for idx in range(len(self)):
            yield self.__class__(
                **{k: v[idx] if v is not None else None
                   for k, v in self.__dict__.items()},
            )

    def __post_init__(self):
        has_state_dist_params = tuple(
            param is not None for param in (self.m, self.V)
        )
        if not len(set(has_state_dist_params)) == 1:
            raise Exception("Provide either all or no distribution parameters")

        has_state_sample = self.x is not None
        if not (all(has_state_dist_params) or has_state_sample):
            raise Exception("Provide at least either dist params or samples.")

        # # checks if all have same time-dimension.
        # # Issue: If used time-wise, e.g. in *_step functions,
        # # then this will be the particle or batch dimension instead.
        # if not len(set(len(v) for v in self.__dict__.values() if v is not None)) == 1:
        #     raise Exception("Not all data in this class has same length.")

@dataclass
class Latents:
    variables: GLSVariables
    gls_params: Optional[Sequence[GLSParams]]

    def __len__(self):
        return len(self.variables)

    def __getitem__(self, item):
        return self.__class__(**{k: v[item] if v is not None else None for k, v in self.__dict__.items()})

    def __iter__(self):
        for idx in range(len(self)):
            yield self.__class__(
                **{k: v[idx] if v is not None else None
                   for k, v in self.__dict__.items()},
            )


@dataclass
class Prediction:
    """
    Template for outputs of all our GLS-based models.
    Note that the latents and emission here have
    1 more dimension in all corresponding tensors, i.e. time-dimension.
    """

    latents: Latents
    emissions: (torch.Tensor, torch.distributions.Distribution)

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, item):
        return self.__class__(**{k: v[item] if v is not None else None for k, v in self.__dict__.items()})

    def __iter__(self):
        for idx in range(len(self)):
            yield self.__class__(
                **{k: v[idx] if v is not None else None
                   for k, v in self.__dict__.items()},
            )


class BaseGaussianLinearSystem(nn.Module, metaclass=ABCMeta):
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
        past_targets_is_observed: Optional[
                Union[Sequence[torch.Tensor], torch.Tensor]] = None,
    ) -> Sequence[Latents]:
        raise NotImplementedError("Should be implemented by child class")

    def smooth(
        self,
        past_targets: [Sequence[torch.Tensor], torch.Tensor],
        past_controls: Optional[Union[Sequence[ControlInputs], ControlInputs]] = None,
        past_targets_is_observed: Optional[
                Union[Sequence[torch.Tensor], torch.Tensor]] = None,
    ) -> Sequence[Latents]:
        raise NotImplementedError("Should be implemented by child class")

    def sample_generative(
        self,
        n_steps_forecast: int,
        n_batch: int,
        future_controls: Optional[Union[Sequence[torch.Tensor], ControlInputs]],
        deterministic: bool = False,
        **kwargs,
    ) -> Sequence[Prediction]:
        raise NotImplementedError("Should be implemented by child class")

    def forecast(
        self,
        n_steps_forecast: int,
        initial_latent: Latents,
        future_controls: Optional[Union[Sequence[ControlInputs], ControlInputs]],
        deterministic: bool = False,
    ) -> Sequence[Prediction]:
        raise NotImplementedError("Should be implemented by child class")

    def predict(
        self,
        n_steps_forecast: int,
        past_targets: [Sequence[torch.Tensor], torch.Tensor],
        past_controls: Optional[Union[Sequence[ControlInputs], ControlInputs]],
        future_controls: Optional[Union[Sequence[ControlInputs], ControlInputs]],
        past_targets_is_observed: Optional[
            Union[Sequence[torch.Tensor], torch.Tensor]] = None,
        deterministic: bool = False,
    ) -> Sequence[Prediction]:
        raise NotImplementedError("Should be implemented by child class")

    def loss(
        self,
        past_targets: [Sequence[torch.Tensor], torch.Tensor],
        past_controls: Optional[Union[Sequence[ControlInputs], ControlInputs]] = None,
        past_targets_is_observed: Optional[
            Union[Sequence[torch.Tensor], torch.Tensor]] = None,
    ) -> torch.Tensor:
        raise NotImplementedError("Should be implemented by child class")
