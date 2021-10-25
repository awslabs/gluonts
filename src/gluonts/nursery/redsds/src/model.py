from typing import Dict, Optional, Tuple
import torch
import torch.distributions as td
import numpy as np
from torch import nn

from .utils import normalize_logprob, get_precision
from .torch_utils import unravel_indices, torch2numpy
from .forward_backward import forward_backward_hmm, forward_backward_hsmm


class Base(nn.Module):
    def __init__(
        self,
        x_init: nn.Module,
        continuous_transition_network: nn.Module,
        z_init: nn.Module,
        discrete_transition_network: nn.Module,
        emission_network: nn.Module,
        inference_network: nn.Module,
        ctrl_transformer: nn.Module,
        continuous_state_dim: int,
        num_categories: int,
        context_length: int,
        prediction_length: int,
        discrete_state_prior: Optional[torch.Tensor] = None,
        transform_target: bool = False,
        transform_only_scale: bool = False,
        use_jacobian: bool = False
    ):
        """Base Model class.

        Args:
            x_init (nn.Module):
                A torch module that models p(x[1] | z[1]).
            continuous_transition_network (nn.Module):
                A torch module that models p(x[t] | x[t-1], z[t]).
            z_init (nn.Module):
                A torch module that models p(z[1]).
            discrete_transition_network (nn.Module):
                A torch module that models p(z[t] | z[t-1], y[t-1], x[t-1]).
            emission_network (nn.Module):
                A torch module that models p(y[t] | x[t]).
            inference_network (nn.Module):
                A torch module that models, q(x[1:T] | y[1:T]), the variational
                distribution of the continuous state.
            ctrl_transformer (nn.Module):
                A network that transforms raw control inputs into
                a feature vector.
            continuous_state_dim (int):
                Dimension of the continuous state x.
            num_categories (int):
                Number of discrete states z.
            context_length (int):
                Context length.
            prediction_length (int):
                Prediction length.
            discrete_state_prior (torch.Tensor, optional):
                Prior on the discrete state. Used for cross-entropy
                regularization. A tensor of probabilities.
                Defaults to None.
            transform_target (bool):
                Apply affine transformation to target.
            transform_only_scale (bool):
                Transform target only by scaling.
            use_jacobian (bool)
                Add the jacobian of the target transformation
                in the likelihood computation.
        """
        super().__init__()

        self.context_length = context_length
        self.prediction_length = prediction_length
        self.K = num_categories
        self.x_dim = continuous_state_dim

        # Submodules
        self.ctrl_transformer = ctrl_transformer
        self.x_init = x_init
        self.x_tran = continuous_transition_network
        self.z_init = z_init
        self.z_tran = discrete_transition_network
        self.y_emit = emission_network
        self.inference_network = inference_network
        self.transform_target = transform_target
        self.transform_only_scale = transform_only_scale
        self.use_jacobian = use_jacobian

        if discrete_state_prior is None:
            self.register_buffer(
                'discrete_prior', torch.ones([self.K]) / self.K)
        else:
            self.register_buffer(
                'discrete_prior', discrete_state_prior)

    def _calculate_continuous_transition_probabilities(
            self,
            x_samples: torch.Tensor,
            ctrl_feats: Optional[torch.Tensor],
            log_prob_x0: torch.Tensor):
        #  Feed x[1:T-1] into the continuous state transition
        #  network to get the distributions:
        #  p(x[2] | x[1], z[2], u[2]) ... p(x[T] | x[T-1], z[T], u[T]).
        #  where u[t] is the control at t-th time step.
        p_xt_dists = self.x_tran(
            x_samples[:, :-1, :],
            ctrl_feats[:, 1:, :] if ctrl_feats is not None else None)
        future_xts = x_samples[:, 1:, :]
        #  Evaluate the log_prob of x[2] ... x[T].
        log_prob_xt = p_xt_dists.log_prob(future_xts[:, :, None, :])
        #  Concatenate the log_prob of x[1] to the log_probs of x[2] ... x[T].
        log_prob_xt = torch.cat([log_prob_x0, log_prob_xt], dim=1)
        return log_prob_xt

    def _calculate_discrete_transition_probabilities(
            self,
            y: torch.Tensor,
            x: torch.Tensor,
            ctrl_feats: Optional[torch.Tensor],
            temperature: float = 1.0):
        B, T = y.shape[:2]
        #  Feed the observations y[1:T-1] and the pseudo observations x[1:T-1]
        #  into the discrete state transition network
        #  to get the distributions:
        #  p(z[2] | x[1], y[1], z[1]) ... p(z[T] | x[T-1], y[T-1], z[T-1]).
        log_prob_zt_ztm1 = self.z_tran(
            y[:, :-1, :],
            x[:, :-1, :],
            ctrl_feats[:, 1:, :] if ctrl_feats is not None else None
        ).view([B, T-1, self.K, self.K])
        #  self.z_tran returns an unnormalized transition matrix,
        #  so we normalize the last axis such that
        #  log_prob_zt_ztm1[:, t, i, j] represents the probability
        #  p(z[t] = j | x[t-1], y[t-1], z[t-1] = i).
        log_prob_zt_ztm1 = normalize_logprob(
            log_prob_zt_ztm1, axis=-1, temperature=temperature)[0]
        identity_pad = torch.eye(self.K, device=y.device)\
            .view(1, 1, self.K, self.K).repeat(B, 1, 1, 1) + get_precision(y)
        log_identity_pad = identity_pad.log()
        #  Pad the [:, 0, :, :] elements with log I. This is to ensure
        #  correct dimensions and the 0-th index is not used in computations.
        log_prob_zt_ztm1 = torch.cat(
            [log_identity_pad, log_prob_zt_ztm1], dim=1)
        return log_prob_zt_ztm1

    def _calculate_likelihoods(
            self,
            y: torch.Tensor,
            x_samples: torch.Tensor,
            scale: torch.Tensor,
            ctrl_feats: Optional[torch.Tensor],):
        B, T = y.shape[:2]
        x0_samples = x_samples[:, 0, :]
        #  Construct the initial x distribution
        if ctrl_feats is None:
            #  If the dataset has no controls,
            #  then pass a tensor of ones as dummy controls.
            dummy_ctrls = torch.ones(B, 1, device=y.device)
            x0_dist = self.x_init(dummy_ctrls)
        else:
            x0_dist = self.x_init(ctrl_feats[:, 0, :])
        #  Compute the log prob p(x[1] | z[1])
        log_prob_x0 = x0_dist.log_prob(x0_samples[:, None, :])
        log_prob_x0 = log_prob_x0[:, None, :]
        #  Compute the log probs p(x[2] | x[1], z[2]) ...
        #  p(x[T] | x[T-1], z[T]).
        log_prob_xt = self._calculate_continuous_transition_probabilities(
            x_samples, ctrl_feats, log_prob_x0)

        #  Compute the log probs p(y[1] | x[1]) ... p(y[T] | x[T]).
        emission_dist = self.y_emit(x_samples)
        log_prob_yt = emission_dist.log_prob(y.view(B, T, -1))

        # Compute Jacobian
        if self.transform_target and self.use_jacobian:
            jac = torch.log(torch.abs(scale)).reshape(-1, 1)
            log_prob_yt = log_prob_yt - jac

        #  Add the log probs p(y[t] | x[t]) and p(x[t] | x[t-1], z[t])
        #  to get log p(y[t], x[t] | x[t-1], z[t]).
        log_p_xt_yt = log_prob_yt[:, :, None] + log_prob_xt
        return log_p_xt_yt

    def get_reconstruction(
            self,
            x_samples: torch.Tensor):
        emission_dist = self.y_emit(x_samples)
        reconstructed_obs = emission_dist.mean
        return reconstructed_obs

    def forward(
        self,
        y: torch.Tensor,
        ctrl_inputs: Optional[Dict[str, torch.Tensor]] = None,
        switch_temperature: float = 1.,
        cont_ent_anneal: float = 1.,
        num_samples: int = 1,
        deterministic_inference: bool = False
    ):
        raise NotImplementedError(
            'This is the Base class. Use one of the child classes.'
        )


class SNLDS(Base):
    def __init__(
        self,
        x_init: nn.Module,
        continuous_transition_network: nn.Module,
        z_init: nn.Module,
        discrete_transition_network: nn.Module,
        emission_network: nn.Module,
        inference_network: nn.Module,
        ctrl_transformer: Optional[nn.Module],
        continuous_state_dim: int,
        num_categories: int,
        context_length: int,
        prediction_length: int,
        discrete_state_prior: Optional[torch.Tensor] = None,
        transform_target: bool = False,
        transform_only_scale: bool = False,
        use_jacobian: bool = False
    ):
        super().__init__(
            x_init,
            continuous_transition_network,
            z_init,
            discrete_transition_network,
            emission_network,
            inference_network,
            ctrl_transformer,
            continuous_state_dim,
            num_categories,
            context_length,
            prediction_length,
            discrete_state_prior=discrete_state_prior,
            transform_target=transform_target,
            transform_only_scale=transform_only_scale,
            use_jacobian=use_jacobian
        )

    def _calculate_loglike_lowerbound(
        self,
        log_a: torch.Tensor,
        log_b: torch.Tensor,
        log_z_init: torch.Tensor,
        log_gamma: torch.Tensor,
        log_xi: torch.Tensor
    ):
        gamma = torch.exp(log_gamma)  # Use .detach() to stop gradient
        xi = torch.exp(log_xi)  # Use .detach() to stop gradient
        log_a = torch.transpose(log_a, -2, -1)
        t2 = torch.sum(xi[:, 1:, :, :] * (log_b[:, 1:, None, :] +
                                          log_a[:, 1:, :, :]), dim=[1, 2, 3])
        gamma_1, log_b1 = gamma[:, 0, :], log_b[:, 0, :]
        t1 = torch.sum(gamma_1 * (log_b1 + log_z_init[:, :]), dim=-1)
        loglike_p_XY_lowerbound = t1 + t2
        return loglike_p_XY_lowerbound

    def _get_iwlbo(
        self,
        loglike_p_XY_lowerbound: torch.Tensor,
        log_prob_q: torch.Tensor,
        num_samples: int
    ):
        log_surrogate_posterior = torch.sum(log_prob_q, dim=-1)

        loglike_p_XY_lowerbound = loglike_p_XY_lowerbound.view(
            [num_samples, -1])
        log_surrogate_posterior = log_surrogate_posterior.view(
            [num_samples, -1])

        iwlbo = torch.logsumexp(
            loglike_p_XY_lowerbound - log_surrogate_posterior, dim=0) -\
            np.log(num_samples)
        iwlbo = iwlbo.mean(0)
        return iwlbo

    def _calculate_objective(
        self,
        log_a: torch.Tensor,
        log_b: torch.Tensor,
        log_z_init: torch.Tensor,
        log_gamma: torch.Tensor,
        log_xi: torch.Tensor,
        log_prob_q: torch.Tensor,
        x_entropy: torch.Tensor,
        log_p_XY: torch.Tensor,
        num_samples: int
    ):
        loglike_p_XY_lowerbound = self._calculate_loglike_lowerbound(
            log_a, log_b, log_z_init, log_gamma, log_xi)
        entropy_term = torch.sum(x_entropy, dim=1).mean(0)
        #  likelihood_termv1 is the lower-bound on p(X, Y) proposed by
        #  Dong et. al, 2020.
        likelihood_termv1 = loglike_p_XY_lowerbound.mean(0)
        #  likelihood_termv2 is p(X, Y) computed using the forward algorithm.
        likelihood_termv2 = log_p_XY.mean(0)
        elbo = likelihood_termv1 + entropy_term
        elbov2 = likelihood_termv2 + entropy_term
        iwlbo = self._get_iwlbo(
            loglike_p_XY_lowerbound, log_prob_q, num_samples)
        iwlbov2 = self._get_iwlbo(
            log_p_XY, log_prob_q, num_samples)
        return elbo, elbov2, iwlbo, iwlbov2

    def forward(
        self,
        y: torch.Tensor,
        ctrl_inputs: Optional[Dict[str, torch.Tensor]] = None,
        switch_temperature: float = 1.,
        cont_ent_anneal: float = 1.,
        num_samples: int = 1,
        deterministic_inference: bool = False
    ):
        y = y[:, :self.context_length, :]
        eps = get_precision(y)
        #  Scale and shift input
        if self.transform_target:
            with torch.no_grad():
                if self.transform_only_scale:
                    scale_y = torch.mean(
                        torch.abs(y), dim=-2, keepdim=True) + eps
                    target_transformer = td.transforms.AffineTransform(
                        torch.zeros_like(scale_y), scale_y
                    )
                else:
                    mu_y = torch.mean(y, dim=-2, keepdim=True)
                    std_y = torch.std(y, dim=-2, keepdim=True) + eps
                    target_transformer = td.transforms.AffineTransform(
                        mu_y, std_y
                    )
                y = target_transformer.inv(y)
        else:
            tmp = torch.mean(y, dim=-2, keepdim=True)
            target_transformer = td.transforms.AffineTransform(
                torch.zeros_like(tmp), torch.ones_like(tmp)
            )

        #  Extract the control features
        ctrl_feats = None
        if self.ctrl_transformer is not None:
            assert ctrl_inputs is not None,\
                'ctrl_inputs cannot be None for a model with ctrl_transformer!'
            ctrl_feats = self.ctrl_transformer(
                feat_static=ctrl_inputs['feat_static_cat'],
                n_timesteps=self.context_length,
                feat_time=ctrl_inputs['past_time_feat'][
                    ..., :self.context_length, :])
        #  Feed the observations y[1:T] into the inference_network
        #  to get the samples from q(x[1:T] | y[1:T]), its entropy
        #  and the log_prob of the samples under q(x[1:T] | y[1:T]).
        x_samples, x_entropy, log_prob_q = self.inference_network(
            y, ctrl_feats=ctrl_feats, num_samples=num_samples,
            deterministic=deterministic_inference)
        _, B, T, x_dim = x_samples.shape

        #  Merge the first two dimensions into a single batch dimension
        x_samples = x_samples.view(num_samples * B, T, x_dim)
        x_entropy = x_entropy.view(num_samples * B, T)
        log_prob_q = log_prob_q.view(num_samples * B, T)

        #  Repeat the first dim num_samples times to allow broadcast
        #  with x_samples.
        y_tiled = y.repeat(num_samples, 1, 1)
        ctrl_feats_tiled = None
        if self.ctrl_transformer is not None:
            ctrl_feats_tiled = ctrl_feats.repeat(num_samples, 1, 1)
        #  Construct the initial discrete distribution p(z[1]).
        if ctrl_feats_tiled is None:
            #  Use a tensor of ones as dummy control if the dataset
            #  does not provide controls.
            dummy_ctrls = torch.ones(y_tiled.size(0), 1, device=y.device)
            log_z_init = self.z_init(dummy_ctrls)
        else:
            log_z_init = self.z_init(ctrl_feats_tiled[:, 0, :])
        #  Compute log B, B = p(y[t], x[t] | x[t-1], z[t]).
        log_b = self._calculate_likelihoods(
            y_tiled,
            x_samples,
            target_transformer.scale.repeat(num_samples, 1, 1),
            ctrl_feats_tiled)
        #  Compute log A, A = p(z[t] | x[t-1], y[t-1], z[t-1]).
        log_a = self._calculate_discrete_transition_probabilities(
            y_tiled, x_samples, ctrl_feats_tiled,
            temperature=switch_temperature)
        #  Compute gamma, xi, and, log p(X, Y) using the forward-backward
        #  algorithm.
        _, _, log_gamma, log_xi, log_p_XY = forward_backward_hmm(
            log_a, log_b, log_z_init)
        #  Compute the objective function.
        elbo, elbov2, iwlbo, iwlbov2 = self._calculate_objective(
            log_a, log_b, log_z_init, log_gamma, log_xi,
            log_prob_q, cont_ent_anneal * x_entropy, log_p_XY, num_samples
        )
        #  Reconstruct the observations for visualization.
        recons_y = self.get_reconstruction(x_samples).view([
            num_samples, B, T, -1])
        #   Compute the KL between the discrete prior and gamma.
        crossent_regularizer = torch.einsum(
            'ijk, k -> ij',
            log_gamma,
            self.discrete_prior
        ).sum(1).mean(0)
        log_gamma = log_gamma.view([num_samples, B, T, -1])
        x_samples = x_samples.view([num_samples, B, T, x_dim])
        # Invert scale and shift
        if self.transform_target:
            with torch.no_grad():
                y = target_transformer(y)
                recons_y = target_transformer(recons_y)

        return_dict = dict(
            elbo=elbo,
            iwlbo=iwlbo,
            elbov2=elbov2,
            iwlbov2=iwlbov2,
            inputs=y,
            reconstructions=recons_y[0],
            x_samples=x_samples[0],
            log_gamma=log_gamma,
            crossent_regularizer=crossent_regularizer
        )
        return return_dict

    def _unroll(
        self,
        start_state: Tuple[torch.Tensor, torch.Tensor],
        T: int,
        future_ctrl_feats: Optional[torch.Tensor],
        deterministic_z: bool = False,
        deterministic_x: bool = False,
        deterministic_y: bool = False,
        drop_first: bool = False
    ):
        z0, x0 = start_state
        num_samples, _ = x0.shape
        y_samples = []
        z_samples = []
        zt, xt = z0, x0
        for i in range(T):
            xt = xt.unsqueeze(1)
            if deterministic_y:
                yt = self.y_emit(xt).mean.squeeze(1)
            else:
                yt = self.y_emit(xt).sample().squeeze(1)
            y_samples.append(yt)
            ctrl_feat_tp1 = None
            if future_ctrl_feats is not None:
                ctrl_feat_tp1 = future_ctrl_feats[:, i:i+1, :]
            yt = yt.unsqueeze(1)
            ztp1_prob = normalize_logprob(
                self.z_tran(yt, xt, ctrl_feat_tp1)[
                    torch.arange(0, num_samples),
                    0, zt, :], axis=-1)[0].exp()
            ztp1_dist = td.categorical.Categorical(probs=ztp1_prob)
            if deterministic_z:
                ztp1 = torch.argmax(ztp1_dist.probs, dim=-1)
            else:
                ztp1 = ztp1_dist.sample()
            xtp1_dist = self.x_tran(xt, ctrl_feat_tp1)
            if deterministic_x:
                xtp1 = xtp1_dist.mean[
                    torch.arange(0, num_samples), 0, ztp1, :
                ]
            else:
                xtp1 = xtp1_dist.sample()[
                    torch.arange(0, num_samples), 0, ztp1, :
                ]
            zt = ztp1
            xt = xtp1
            z_samples.append(zt)
        z_samples = torch.stack(z_samples).permute(1, 0)
        y_samples = torch.stack(y_samples).permute(1, 0, 2)
        if drop_first:
            xt = xt.unsqueeze(1)
            if deterministic_y:
                yt = self.y_emit(xt).mean.squeeze(1)
            else:
                yt = self.y_emit(xt).sample().squeeze(1)
            y_samples = torch.cat([y_samples, yt[:, None, :]], dim=-2)
            y_samples = y_samples[:, 1:, :]
        return y_samples, z_samples

    @torch.no_grad()
    def predict(
        self,
        y: torch.Tensor,
        ctrl_inputs: Optional[Dict[str, torch.Tensor]] = None,
        num_samples: int = 1,
        deterministic_z: bool = False,
        deterministic_x: bool = False,
        deterministic_y: bool = False,
        mean_prediction: bool = False
    ):
        if mean_prediction:
            num_samples = 1
        self.eval()
        y = y[..., :self.context_length, :]
        eps = get_precision(y)
        # Scale and shift input
        if self.transform_target:
            with torch.no_grad():
                if self.transform_only_scale:
                    scale_y = torch.mean(
                        torch.abs(y), dim=-2, keepdim=True) + eps
                    target_transformer = td.transforms.AffineTransform(
                        torch.zeros_like(scale_y), scale_y
                    )
                else:
                    mu_y = torch.mean(y, dim=-2, keepdim=True)
                    std_y = torch.std(y, dim=-2, keepdim=True) + eps
                    target_transformer = td.transforms.AffineTransform(
                        mu_y, std_y
                    )
                y = target_transformer.inv(y)
        else:
            tmp = torch.mean(y, dim=-2, keepdim=True)
            target_transformer = td.transforms.AffineTransform(
                torch.zeros_like(tmp), torch.ones_like(tmp)
            )

        # Extract the control features
        ctrl_feats = None
        if self.ctrl_transformer is not None:
            assert ctrl_inputs is not None,\
                'ctrl_inputs cannot be None for a model with ctrl_transformer!'
            ctrl_feats = self.ctrl_transformer(
                feat_static=ctrl_inputs['feat_static_cat'],
                n_timesteps=self.context_length,
                feat_time=ctrl_inputs['past_time_feat'])
        #  Infer the latent state x[1:T]
        x_samples, _, _ = self.inference_network(
            y, ctrl_feats, num_samples=num_samples,
            deterministic=mean_prediction)
        _, B, T, x_dim = x_samples.shape
        x_samples = x_samples.view(num_samples * B, T, x_dim)
        #  Repeat the first dim num_samples times to allow broadcast
        #  with x_samples.
        y_tiled = y.repeat(num_samples, 1, 1)
        ctrl_feats_tiled = None
        if self.ctrl_transformer is not None:
            ctrl_feats_tiled = ctrl_feats.repeat(num_samples, 1, 1)
        #  Construct the initial discrete distribution p(z[1]).
        if ctrl_feats_tiled is None:
            #  Use a tensor of ones as dummy control if the dataset
            #  does not provide controls.
            dummy_ctrls = torch.ones(y_tiled.size(0), 1, device=y.device)
            log_z_init = self.z_init(dummy_ctrls)
        else:
            log_z_init = self.z_init(ctrl_feats_tiled[:, 0, :])
        #  Compute log B, B = p(y[t], x[t] | x[t-1], z[t]).
        log_b = self._calculate_likelihoods(
            y_tiled,
            x_samples,
            target_transformer.scale.repeat(num_samples, 1, 1),
            ctrl_feats_tiled)
        #  Compute log A, A = p(z[t] | y[t-1], z[t-1]).
        log_a = self._calculate_discrete_transition_probabilities(
            y_tiled, x_samples, ctrl_feats_tiled, temperature=1.)
        #  Compute gamma using the forward-backward
        #  algorithm.
        _, _, log_gamma, _, _ = forward_backward_hmm(
            log_a, log_b, log_z_init)
        #  Reconstruct the input
        rec_y = self.get_reconstruction(x_samples).view([
            num_samples, B, T, -1])
        #  Get the most likely state for zT
        zT = torch.argmax(log_gamma[:, -1, :], dim=-1)
        #  Get the state xT
        xT = x_samples[:, -1, :]
        #  Get future control features
        future_ctrl_feats = None
        if self.ctrl_transformer is not None:
            future_ctrl_feats = self.ctrl_transformer(
                feat_static=ctrl_inputs['feat_static_cat'],
                n_timesteps=self.prediction_length,
                feat_time=ctrl_inputs['future_time_feat']).repeat(
                    num_samples, 1, 1)
        #  Unroll using zT and xT
        forecast, z_samples = self._unroll(
            start_state=(zT, xT),
            T=self.prediction_length,
            future_ctrl_feats=future_ctrl_feats,
            deterministic_z=deterministic_z,
            deterministic_x=deterministic_x,
            deterministic_y=deterministic_y,
            drop_first=True)
        forecast = forecast.view(
            [num_samples, B, self.prediction_length, -1])
        z_samples = z_samples.view(
            [num_samples, B, self.prediction_length, 1])
        z_samples_oh = torch.zeros(
            num_samples, B, self.prediction_length, self.K,
            device=z_samples.device)
        z_samples_oh.scatter_(-1, z_samples, 1)
        z_emp_probs = z_samples_oh.mean(0)
        #  Concat reconstruction with forecast on time dimension
        rec_y_with_forecast = torch.cat([rec_y, forecast], dim=-2)
        # Invert scale and shift
        if self.transform_target:
            with torch.no_grad():
                rec_y_with_forecast = target_transformer(rec_y_with_forecast)
        return dict(
            rec_n_forecast=rec_y_with_forecast,
            z_emp_probs=z_emp_probs
        )


class REDSDS(Base):
    def __init__(
        self,
        x_init: nn.Module,
        continuous_transition_network: nn.Module,
        z_init: nn.Module,
        discrete_transition_network: nn.Module,
        emission_network: nn.Module,
        inference_network: nn.Module,
        ctrl_transformer: Optional[nn.Module],
        ctrl2nstf_network: nn.Module,
        continuous_state_dim: int,
        num_categories: int,
        d_max: int,
        context_length: int,
        prediction_length: int,
        discrete_state_prior: Optional[torch.Tensor] = None,
        transform_target: bool = False,
        transform_only_scale: bool = False,
        use_jacobian: bool = False
    ):
        """REDSDS Model class.

        Args:
            x_init (nn.Module):
                A torch module that models p(x[1] | z[1]).
            continuous_transition_network (nn.Module):
                A torch module that models p(x[t] | x[t-1], z[t]).
            z_init (nn.Module):
                A torch module that models p(z[1]).
            discrete_transition_network (nn.Module):
                A torch module that models p(z[t] | z[t-1], y[t-1], x[t-1]).
            emission_network (nn.Module):
                A torch module that models p(y[t] | x[t]).
            inference_network (nn.Module):
                A torch module that models, q(x[1:T] | y[1:T]), the variational
                distribution of the continuous state.
            ctrl_transformer (nn.Module):
                A network that transforms control variables into
                a feature vector.
            ctrl2nstf_network (nn.Module):
                A network that takes in control variables and gives
                the non-stationary transition functions that model
                the durations.
            continuous_state_dim (int):
                Dimension of the continuous state x.
            num_categories (int):
                Number of discrete states z.
            d_max (int):
                Maximum duration of discrete state.
            context_length (int):
                Context length.
            prediction_length (int):
                Prediction length.
            discrete_state_prior (torch.Tensor, optional):
                Prior on the discrete state. Used for cross-entropy
                regularization. A tensor of probabilities.
                Defaults to None.
            transform_target (bool):
                Apply affine transformation to target.
            transform_only_scale (bool):
                Transform target only by scaling.
            use_jacobian (bool)
                Add the jacobian of the target transformation
                in the likelihood computation.
        """
        super().__init__(
            x_init,
            continuous_transition_network,
            z_init,
            discrete_transition_network,
            emission_network,
            inference_network,
            ctrl_transformer,
            continuous_state_dim,
            num_categories,
            context_length,
            prediction_length,
            discrete_state_prior=discrete_state_prior,
            transform_target=transform_target,
            transform_only_scale=transform_only_scale,
            use_jacobian=use_jacobian
        )
        self.d_max = d_max
        self.ctrl2nstf_network = ctrl2nstf_network

    def _get_iwlbo(
        self,
        loglike_p_XY: torch.Tensor,
        log_prob_q: torch.Tensor,
        num_samples: int
    ):
        log_surrogate_posterior = torch.sum(log_prob_q, dim=-1)

        loglike_p_XY = loglike_p_XY.view(
            [num_samples, -1])
        log_surrogate_posterior = log_surrogate_posterior.view(
            [num_samples, -1])

        iwlbo = torch.logsumexp(
            loglike_p_XY - log_surrogate_posterior, dim=0) -\
            np.log(num_samples)
        iwlbo = iwlbo.mean(0)
        return iwlbo

    def _calculate_objective(
        self,
        log_a: torch.Tensor,
        log_b: torch.Tensor,
        log_z_init: torch.Tensor,
        log_gamma: torch.Tensor,
        log_prob_q: torch.Tensor,
        x_entropy: torch.Tensor,
        log_p_XY: torch.Tensor,
        num_samples: int
    ):
        entropy_term = torch.sum(x_entropy, dim=1).mean(0)
        #  likelihood_termv2 is p(X, Y) computed using the forward algorithm.
        likelihood_termv2 = log_p_XY.mean(0)
        elbov2 = likelihood_termv2 + entropy_term
        iwlbov2 = self._get_iwlbo(
            log_p_XY, log_prob_q, num_samples)
        return elbov2, iwlbov2

    def forward(
        self,
        y: torch.Tensor,
        ctrl_inputs: Optional[Dict[str, torch.Tensor]] = None,
        switch_temperature: float = 1.,
        dur_temperature: float = 1.,
        cont_ent_anneal: float = 1.,
        num_samples: int = 1,
        deterministic_inference: bool = False
    ):
        y = y[:, :self.context_length, :]
        eps = get_precision(y)
        # Scale and shift input
        if self.transform_target:
            with torch.no_grad():
                if self.transform_only_scale:
                    scale_y = torch.mean(
                        torch.abs(y), dim=-2, keepdim=True) + eps
                    target_transformer = td.transforms.AffineTransform(
                        torch.zeros_like(scale_y), scale_y
                    )
                else:
                    mu_y = torch.mean(y, dim=-2, keepdim=True)
                    std_y = torch.std(y, dim=-2, keepdim=True) + eps
                    target_transformer = td.transforms.AffineTransform(
                        mu_y, std_y
                    )
                y = target_transformer.inv(y)
        else:
            tmp = torch.mean(y, dim=-2, keepdim=True)
            target_transformer = td.transforms.AffineTransform(
                torch.zeros_like(tmp), torch.ones_like(tmp)
            )

        #  Extract the control features
        ctrl_feats = None
        if self.ctrl_transformer is not None:
            assert ctrl_inputs is not None,\
                'ctrl_inputs cannot be None for a model with ctrl_transformer!'
            ctrl_feats = self.ctrl_transformer(
                feat_static=ctrl_inputs['feat_static_cat'],
                n_timesteps=self.context_length,
                feat_time=ctrl_inputs['past_time_feat'][
                    ..., :self.context_length, :])
        #  Feed the observations y[1:T] into the inference_network
        #  to get the samples from q(x[1:T] | y[1:T]), it entropy
        #  and the log_prob of the samples under q(x[1:T] | y[1:T]).
        x_samples, x_entropy, log_prob_q = self.inference_network(
            y, ctrl_feats=ctrl_feats, num_samples=num_samples,
            deterministic=deterministic_inference)
        _, B, T, x_dim = x_samples.shape

        #  Merge the first two dimensions into a single batch dimension
        x_samples = x_samples.view(num_samples * B, T, x_dim)
        x_entropy = x_entropy.view(num_samples * B, T)
        log_prob_q = log_prob_q.view(num_samples * B, T)

        #  Repeat the first dim num_samples times to allow broadcast
        #  with x_samples.
        y_tiled = y.repeat(num_samples, 1, 1)
        ctrl_feats_tiled = None
        if self.ctrl_transformer is not None:
            ctrl_feats_tiled = ctrl_feats.repeat(num_samples, 1, 1)
        #  Construct the initial discrete distribution p(z[1]).
        if ctrl_feats_tiled is None:
            #  Use a tensor of ones as dummy control if the dataset
            #  does not provide controls.
            dummy_ctrls = torch.ones(y_tiled.size(0), 1, device=y.device)
            log_z_init = self.z_init(dummy_ctrls)
        else:
            log_z_init = self.z_init(ctrl_feats_tiled[:, 0, :])
        #  Compute log B, B = p(y[t], x[t] | x[t-1], z[t]).
        log_b = self._calculate_likelihoods(
            y_tiled,
            x_samples,
            target_transformer.scale.repeat(num_samples, 1, 1),
            ctrl_feats_tiled)
        #  Compute log A, A = p(z[t] | x[t-1], y[t-1], z[t-1]).
        log_a = self._calculate_discrete_transition_probabilities(
            y_tiled, x_samples, ctrl_feats_tiled,
            temperature=switch_temperature)
        #  Get the NSTFs
        if ctrl_feats_tiled is None:
            #  If the dataset is control-less,
            #  then feed in dummy control = 1
            dummy_ctrls = torch.ones(
                num_samples * B, T, 1, device=y.device)
            log_u = self.ctrl2nstf_network(
                ctrl_feats=dummy_ctrls,
                temperature=dur_temperature)
        else:
            log_u = self.ctrl2nstf_network(
                ctrl_feats=ctrl_feats_tiled,
                temperature=dur_temperature)
        #  Compute gamma, xi, and, log p(X, Y) using the forward-backward
        #  algorithm.
        _, _, log_gamma, log_p_XY = forward_backward_hsmm(
            log_a, log_b, log_z_init, log_u)
        #  State posterior: sum over counts
        log_z_posterior = torch.logsumexp(log_gamma, dim=-1)
        #  Compute the objective function.
        elbov2, iwlbov2 = self._calculate_objective(
            log_a, log_b, log_z_init, log_gamma,
            log_prob_q, cont_ent_anneal * x_entropy, log_p_XY, num_samples)
        #  Reconstruct the observations for visualization.
        recons_y = self.get_reconstruction(x_samples).view([
            num_samples, B, T, -1])
        #   Compute the KL between the discrete prior and gamma.
        crossent_regularizer = torch.einsum(
            'ijk, k -> ij',
            log_z_posterior,
            self.discrete_prior
        ).sum(1).mean(0)
        log_z_posterior = log_z_posterior.view([num_samples, B, T, -1])
        x_samples = x_samples.view([num_samples, B, T, x_dim])
        # Invert scale and shift
        if self.transform_target:
            with torch.no_grad():
                y = target_transformer(y)
                recons_y = target_transformer(recons_y)

        return_dict = dict(
            elbov2=elbov2,
            iwlbov2=iwlbov2,
            inputs=y,
            reconstructions=recons_y[0],
            x_samples=x_samples[0],
            log_gamma=log_z_posterior,
            crossent_regularizer=crossent_regularizer
        )
        return return_dict

    def _unroll(
        self,
        start_state: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        T: int,
        future_ctrl_feats: Optional[torch.Tensor],
        deterministic_z: bool = False,
        deterministic_x: bool = False,
        deterministic_y: bool = False,
        drop_first: bool = False
    ):
        c0, z0, x0 = start_state
        num_samples, _ = x0.shape
        y_samples = []
        z_samples = []
        ct, zt, xt = c0, z0, x0
        for i in range(T):
            xt = xt.unsqueeze(1)
            if deterministic_y:
                yt = self.y_emit(xt).mean.squeeze(1)
            else:
                yt = self.y_emit(xt).sample().squeeze(1)
            y_samples.append(yt)
            ctrl_feat_tp1 = None
            if future_ctrl_feats is not None:
                ctrl_feat_tp1 = future_ctrl_feats[:, i:i+1, :]
            if future_ctrl_feats is None:
                #  If the dataset is control-less,
                #  then feed in dummy control = 1
                dummy_ctrls = torch.ones(
                    num_samples, 1, 1, device=x0.device)
                log_u = self.ctrl2nstf_network(ctrl_feats=dummy_ctrls)
            else:
                log_u = self.ctrl2nstf_network(ctrl_feats=ctrl_feat_tp1)
            u = log_u.exp()[
                torch.arange(0, num_samples),
                0,
                zt.squeeze(),
                ct.squeeze()]
            u_dist = td.bernoulli.Bernoulli(probs=u)
            if deterministic_z:
                u_sample = (u_dist.probs > 0.5).long()
            else:
                u_sample = u_dist.sample().long()
            ctp1 = ct
            if not (
                u_sample.size() == ctp1.size()
                and torch.unique(u_sample).size(0) <= 2
            ):
                print(
                    u_sample.size(),
                    ctp1.size(),
                    torch2numpy(torch.unique(u_sample))
                )
                raise RuntimeError()
            ctp1[u_sample == 1] = ctp1[u_sample == 1] + 1
            ctp1[u_sample == 0] = 0
            yt = yt.unsqueeze(1)
            ztp1_prob = normalize_logprob(
                self.z_tran(yt, xt, ctrl_feat_tp1)[
                    torch.arange(0, num_samples),
                    0, zt, :], axis=-1)[0].exp()
            ztp1_dist = td.categorical.Categorical(probs=ztp1_prob)
            if deterministic_z:
                ztp1 = torch.argmax(ztp1_dist.probs, dim=-1)
            else:
                ztp1 = ztp1_dist.sample()
            ztp1[ctp1 > 0] = zt[ctp1 > 0]

            xtp1_dist = self.x_tran(xt, ctrl_feat_tp1)
            if deterministic_x:
                xtp1 = xtp1_dist.mean[
                    torch.arange(0, num_samples), 0, ztp1, :
                ]
            else:
                xtp1 = xtp1_dist.sample()[
                    torch.arange(0, num_samples), 0, ztp1, :
                ]
            ct = ctp1
            zt = ztp1
            xt = xtp1
            z_samples.append(zt)
        z_samples = torch.stack(z_samples).permute(1, 0)
        y_samples = torch.stack(y_samples).permute(1, 0, 2)
        if drop_first:
            xt = xt.unsqueeze(1)
            if deterministic_y:
                yt = self.y_emit(xt).mean.squeeze(1)
            else:
                yt = self.y_emit(xt).sample().squeeze(1)
            y_samples = torch.cat([y_samples, yt[:, None, :]], dim=-2)
            y_samples = y_samples[:, 1:, :]
        return y_samples, z_samples

    @torch.no_grad()
    def predict(
        self,
        y: torch.Tensor,
        ctrl_inputs: Optional[Dict[str, torch.Tensor]] = None,
        num_samples: int = 1,
        deterministic_z: bool = False,
        deterministic_x: bool = False,
        deterministic_y: bool = False,
        mean_prediction: bool = False
    ):
        if mean_prediction:
            num_samples = 1
        self.eval()
        y = y[..., :self.context_length, :]
        eps = get_precision(y)
        # Scale and shift input
        if self.transform_target:
            with torch.no_grad():
                if self.transform_only_scale:
                    scale_y = torch.mean(
                        torch.abs(y), dim=-2, keepdim=True) + eps
                    target_transformer = td.transforms.AffineTransform(
                        torch.zeros_like(scale_y), scale_y
                    )
                else:
                    mu_y = torch.mean(y, dim=-2, keepdim=True)
                    std_y = torch.std(y, dim=-2, keepdim=True) + eps
                    target_transformer = td.transforms.AffineTransform(
                        mu_y, std_y
                    )
                y = target_transformer.inv(y)
        else:
            tmp = torch.mean(y, dim=-2, keepdim=True)
            target_transformer = td.transforms.AffineTransform(
                torch.zeros_like(tmp), torch.ones_like(tmp)
            )

        # Extract the control features
        ctrl_feats = None
        if self.ctrl_transformer is not None:
            assert ctrl_inputs is not None
            ctrl_feats = self.ctrl_transformer(
                feat_static=ctrl_inputs['feat_static_cat'],
                n_timesteps=self.context_length,
                feat_time=ctrl_inputs['past_time_feat'])
        #  Infer the latent state x[1:T]
        x_samples, _, _ = self.inference_network(
            y, ctrl_feats, num_samples=num_samples,
            deterministic=mean_prediction)
        _, B, T, x_dim = x_samples.shape
        x_samples = x_samples.view(num_samples * B, T, x_dim)
        #  Repeat the first dim num_samples times to allow broadcast
        #  with x_samples.
        y_tiled = y.repeat(num_samples, 1, 1)
        ctrl_feats_tiled = None
        if self.ctrl_transformer is not None:
            ctrl_feats_tiled = ctrl_feats.repeat(num_samples, 1, 1)
        if ctrl_feats_tiled is None:
            #  Use a tensor of ones as dummy control if the dataset
            #  does not provide controls.
            dummy_ctrls = torch.ones(y_tiled.size(0), 1, device=y.device)
            log_z_init = self.z_init(dummy_ctrls)
        else:
            log_z_init = self.z_init(ctrl_feats_tiled[:, 0, :])
        #  Compute log B, B = p(y[t], x[t] | x[t-1], z[t]).
        log_b = self._calculate_likelihoods(
            y_tiled,
            x_samples,
            target_transformer.scale.repeat(num_samples, 1, 1),
            ctrl_feats_tiled)
        #  Compute log A, A = p(z[t] | y[t-1], z[t-1]).
        log_a = self._calculate_discrete_transition_probabilities(
            y_tiled, x_samples, ctrl_feats_tiled, temperature=1.)
        #  Get the NSTFs
        if ctrl_feats_tiled is None:
            #  If the dataset is control-less,
            #  then feed in dummy control = 1
            dummy_ctrls = torch.ones(
                num_samples * B, T, 1, device=log_a.device)
            log_u = self.ctrl2nstf_network(ctrl_feats=dummy_ctrls)
        else:
            log_u = self.ctrl2nstf_network(ctrl_feats=ctrl_feats_tiled)
        #  Compute gamma using the forward-backward
        #  algorithm.
        _, _, log_gamma, _ = forward_backward_hsmm(
            log_a, log_b, log_z_init, log_u)
        #  Reconstruct the input
        rec_y = self.get_reconstruction(x_samples).view([
            num_samples, B, T, -1])
        #  Get the most likely state for cT, zT
        log_gamma_flat = log_gamma.view(
            num_samples * B, T, self.K * self.d_max)
        zTcT = unravel_indices(
            torch.argmax(log_gamma_flat[:, -1, :], dim=-1),
            (self.K, self.d_max))
        zT = zTcT[:, 0].view((num_samples * B,))
        cT = zTcT[:, 1].view((num_samples * B,))
        #  Get the state xT
        xT = x_samples[:, -1, :]
        #  Get future control features
        future_ctrl_feats = None
        if self.ctrl_transformer is not None:
            future_ctrl_feats = self.ctrl_transformer(
                feat_static=ctrl_inputs['feat_static_cat'],
                n_timesteps=self.prediction_length,
                feat_time=ctrl_inputs['future_time_feat']).repeat(
                    num_samples, 1, 1)
        #  Unroll using cT, zT and xT
        forecast, z_samples = self._unroll(
            start_state=(cT, zT, xT),
            T=self.prediction_length,
            future_ctrl_feats=future_ctrl_feats,
            deterministic_z=deterministic_z,
            deterministic_x=deterministic_x,
            deterministic_y=deterministic_y,
            drop_first=True)
        forecast = forecast.view(
            [num_samples, B, self.prediction_length, -1])
        z_samples = z_samples.view(
            [num_samples, B, self.prediction_length, 1])
        z_samples_oh = torch.zeros(
            num_samples, B, self.prediction_length, self.K,
            device=z_samples.device)
        z_samples_oh.scatter_(-1, z_samples, 1)
        z_emp_probs = z_samples_oh.mean(0)
        #  Concat reconstruction with forecast on time dimension
        rec_y_with_forecast = torch.cat([rec_y, forecast], dim=-2)
        # Invert scale and shift
        if self.transform_target:
            with torch.no_grad():
                rec_y_with_forecast = target_transformer(rec_y_with_forecast)

        return dict(
            rec_n_forecast=rec_y_with_forecast,
            z_emp_probs=z_emp_probs
        )
