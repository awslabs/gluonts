from typing import Tuple
import math
import numpy as np
import torch
from torch.distributions.categorical import Categorical

from inference.smc.normalize import normalize_log_weights


def repeat_like(
    self: torch.Tensor, target: torch.Tensor, self_dims_in_target: tuple
):
    assert isinstance(self_dims_in_target, (tuple, list, set))
    self_dims_in_target = tuple(
        sorted(
            dim if dim >= 0 else target.ndim + dim
            for dim in self_dims_in_target
        )
    )
    assert len(self_dims_in_target) == self.ndim
    assert all(dim < target.ndim for dim in self_dims_in_target)
    # assert all(target.shape[dim] == self.shape[i] for i, dim in enumerate(self_dims_in_target))

    repeats = [None] * target.ndim
    for dim in range(target.ndim):
        if dim in self_dims_in_target:
            repeats[dim] = 1
        else:
            self = self.unsqueeze(dim=dim)
            repeats[dim] = target.shape[dim]
    return self.repeat(repeats=repeats)


def make_argmax_log_weights(log_weights, dim: int = -2):
    """ sets the largest weight to 0 and all others to -inf """
    is_max = (
        torch.nn.functional.one_hot(
            torch.argmax(log_weights, dim=dim),
            num_classes=log_weights.shape[dim],
        )
        .transpose(dim, -1)
        .to(torch.bool)
    )
    argmaxed_log_weights = torch.where(
        is_max,
        torch.zeros_like(log_weights),
        -np.inf * torch.ones_like(log_weights),
    )
    return argmaxed_log_weights


def categorical_resampling_indices(
    log_norm_weights: torch.Tensor, n_particle: int, dim: int = -2
):
    permute_dims = list(range(log_norm_weights.ndim))
    permute_dims.append(permute_dims.pop(dim))
    ancestor_dist = Categorical(
        logits=log_norm_weights.permute(dims=permute_dims)
    )
    indices = ancestor_dist.sample([n_particle])  # PB
    return indices.to(torch.int64)


def systematic_resampling_indices(
    log_norm_weights: torch.Tensor, n_particle: int, dim: int = -2
):
    dim = dim if dim >= 0 else log_norm_weights.ndim + dim
    noise_shp = list(log_norm_weights.shape)
    noise_shp[dim] = 1
    dtype, device = log_norm_weights.dtype, log_norm_weights.device

    # redundant.
    log_norm_weights = normalize_log_weights(log_norm_weights, dim=dim)
    # Make systematic positions at "dim" and repeat all other dims like in log_norm_weights.
    # This allows for a different number of particles after re-sampling.
    # This is done as follows: New particle dimension is last, old particle dimension replaced
    # by "1", we broadcast everything, and finally swap "1" with the particle dimension (last).
    _ones = torch.ones(
        tuple(
            n if d != dim else 1 for d, n in enumerate(log_norm_weights.shape)
        ),
        dtype=dtype,
        device=device,
    )
    _pos = torch.arange(n_particle, dtype=dtype, device=device) / n_particle
    systematic_positions = (_ones[..., None] * _pos).transpose(dim, -1)[..., 0]

    sampling_noise = (
        torch.rand(noise_shp, dtype=dtype, device=device) / n_particle
    )
    random_positions = systematic_positions + sampling_noise
    weight_positions = torch.cumsum(torch.exp(log_norm_weights), dim=dim)

    # exclude last elem because last cumsum element is analytically 1 but not numerically!
    indices = (
        torch.le(
            weight_positions.unsqueeze(dim + 1 if dim >= 0 else dim),
            # P1B --> broadcast on "1"
            random_positions.unsqueeze(dim if dim >= 0 else dim - 1),
            # 1PB --> broadcast on "1"
        )[:-1]
        .to(log_norm_weights.dtype)
        .sum(dim=0)
    )  # elem [-1] may not be "True". see above.
    return indices.to(torch.int64)


def log_effective_sample_size(log_norm_weights: torch.Tensor, dim: int = -2):
    """ ESS is the inverse of sum of squared normalised importance weights. """
    log_sum_squared_weights = torch.logsumexp(log_norm_weights * 2, dim=dim)
    return -log_sum_squared_weights


def make_criterion_fn_with_ess_threshold(min_ess_ratio: float = 0.5):
    min_ess_ratio = min_ess_ratio if min_ess_ratio is not None else 0.5

    def criterion_fn(log_norm_weights, dim):
        log_ess = log_effective_sample_size(
            log_norm_weights=log_norm_weights, dim=dim
        )
        log_min_particles = torch.log(
            torch.tensor(
                min_ess_ratio * log_norm_weights.shape[dim],
                dtype=log_norm_weights.dtype,
                device=log_norm_weights.device,
            )
        )
        return log_ess < log_min_particles

    return criterion_fn


def resample(
    n_particle: int,
    log_norm_weights: torch.Tensor,
    tensors_to_resample: Tuple[torch.Tensor, ...],
    resampling_indices_fn: callable,
    criterion_fn: callable,
    dim: int = -2,
):
    dim = dim if dim >= 0 else log_norm_weights.ndim + dim
    shp_resampled_weights = tuple(
        n if d != dim else n_particle
        for d, n in enumerate(log_norm_weights.shape)
    )
    uniform_weights = torch.zeros(
        shp_resampled_weights,
        dtype=log_norm_weights.dtype,
        device=log_norm_weights.device,
    ) - math.log(n_particle)

    if n_particle != log_norm_weights.shape[dim]:  # re-sample all
        shp_needs = tuple(
            n for d, n in enumerate(log_norm_weights.shape) if d != dim
        )
        needs_resampling = torch.ones(
            shp_needs, dtype=torch.bool, device=log_norm_weights.device
        )
        resampled_log_norm_weights = uniform_weights
    else:
        needs_resampling = criterion_fn(
            log_norm_weights=log_norm_weights, dim=dim
        )
        resampled_log_norm_weights = torch.where(
            needs_resampling,
            uniform_weights,
            log_norm_weights,
            # keep the previous weights if needs_resampling is False
        )

    idxs_resample = resampling_indices_fn(
        log_norm_weights=log_norm_weights, n_particle=n_particle
    )
    idxs_keep = repeat_like(
        torch.arange(
            n_particle, dtype=idxs_resample.dtype, device=idxs_resample.device
        ),
        idxs_resample,
        self_dims_in_target=(dim,),
    )
    idxs_resample_if_criterion = torch.where(
        needs_resampling, idxs_resample, idxs_keep,
    )
    resampled_tensors = tuple(
        particles.gather(
            dim=dim,
            index=repeat_like(
                idxs_resample_if_criterion,
                particles,
                self_dims_in_target=tuple(
                    range(idxs_resample_if_criterion.ndim)
                ),
            ),
        )
        for particles in tensors_to_resample
    )
    return resampled_log_norm_weights, resampled_tensors
