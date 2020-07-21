from typing import Sequence, Optional, Union

import torch
from torch.distributions import MultivariateNormal

from inference.smc.normalize import normalize_log_weights
from inference.smc.resampling import make_criterion_fn_with_ess_threshold, \
    systematic_resampling_indices, resample, make_argmax_log_weights
from models_new_will_replace.base_amortized_gls import \
    BaseAmortizedGaussianLinearSystem, LatentsRBSMC
from models_new_will_replace.base_gls import ControlInputs


class BaseRBSMCGaussianLinearSystem(BaseAmortizedGaussianLinearSystem):
    def __init__(
        self,
        *args,
        resampling_criterion_fn=make_criterion_fn_with_ess_threshold(0.5),
        resampling_indices_fn: callable = systematic_resampling_indices,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.resampling_criterion_fn = resampling_criterion_fn
        self.resampling_indices_fn = resampling_indices_fn

    def loss(
        self,
        past_targets: [Sequence[torch.Tensor], torch.Tensor],
        past_controls: Optional[Union[Sequence[ControlInputs], ControlInputs]] = None,
    ) -> torch.Tensor:
        return self.loss_filter(
            past_targets=past_targets,
            past_controls=past_controls,
        )

    def loss_filter(
        self,
        past_targets: [Sequence[torch.Tensor], torch.Tensor],
        past_controls: Optional[Union[Sequence[ControlInputs], ControlInputs]] = None,
    ) -> torch.Tensor:
        """
        Computes an estimate of the negative log marginal likelihood.

        Note: the importance weights exp(log_weights) must be un-normalized
        and correspond to the conditional distributions
        (i.e. incremental importance weights / importance weight updates) s.t.
        their product yields an (unbiased) estimate of the marginal likelihood.
        """
        latents_filtered = self.filter(
            past_targets=past_targets, past_controls=past_controls,
        )
        log_weights = [lats.log_weights for lats in latents_filtered]
        log_conditionals = [torch.logsumexp(lws, dim=0) for lws in log_weights]
        log_marginal = sum(log_conditionals)  # FIVO-type ELBO
        return -log_marginal

    def _prepare_forecast(
            self,
            initial_latent: LatentsRBSMC,
            controls: Optional[
                Union[Sequence[ControlInputs], ControlInputs]] = None,
            deterministic: bool = False,
    ):
        cls = initial_latent.variables.__class__

        resampled_log_norm_weights, resampled_tensors = resample(
            n_particle=self.n_particle,
            log_norm_weights=normalize_log_weights(
                log_weights=initial_latent.log_weights
                if not deterministic
                else make_argmax_log_weights(initial_latent.log_weights),
            ),
            tensors_to_resample={
                k: v for k, v in initial_latent.variables.__dict__.items()
                if v is not None
            },
            resampling_indices_fn=self.resampling_indices_fn,
            criterion_fn=make_criterion_fn_with_ess_threshold(
                min_ess_ratio=1.0,  # re-sample always / all.
            ),
        )
        for k, v in initial_latent.variables.__dict__.items():
            if v is None:
                resampled_tensors.update({k: v})

        # pack re-sampled back into object of our API type.
        resampled_initial_latent = initial_latent.__class__(
            log_weights=resampled_log_norm_weights,
            variables=cls(**resampled_tensors, ),
            gls_params=None,  # remember to also re-sample these if need to use.
        )

        if resampled_initial_latent.variables.x is None:
            resampled_initial_latent.variables.x = MultivariateNormal(
                loc=resampled_initial_latent.variables.m,
                covariance_matrix=resampled_initial_latent.variables.V,
            ).rsample()
            resampled_initial_latent.variables.m = None
            resampled_initial_latent.variables.V = None

        return resampled_initial_latent, controls