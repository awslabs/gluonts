import torch
import torch.nn as nn
import torch.distributions as td
import torch.nn.functional as F
import numpy as np

from .utils import (
    normalize_logprob,
    clamp_probabilities,
    inverse_softplus
)


SCALE_OFFSET = 1e-6


class SelectIndex(nn.Module):
    def __init__(self, index=0):
        """Helper module to select the tensor at the given index
        from a tuple of tensors.

        Args:
            index (int, optional):
                The index to select. Defaults to 0.
        """
        super().__init__()
        self.index = index

    def forward(self, x):
        return x[self.index]


class PositionalEncoding(nn.Module):
    def __init__(self, feat_dim, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, feat_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, feat_dim, 2).float()
                             * (-np.log(10000.0) / feat_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        if feat_dim > 1:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        _, B, _ = x.shape
        pe = self.pe[:x.size(0), :].repeat(1, B, 1)
        x = torch.cat([x, pe], dim=-1)
        return x


class TransformerEmbedder(nn.Module):
    def __init__(
        self,
        obs_dim,
        emb_dim=4,
        use_pe=True,
        nhead=1,
        dim_feedforward=32,
        dropout=0.1,
        n_layers=1,
    ):
        super().__init__()
        self.use_pe = use_pe
        self.linear_map = nn.Linear(obs_dim, emb_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=2 * emb_dim if use_pe else emb_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout)
        self.pos_encoder = PositionalEncoding(emb_dim) if use_pe\
            else None
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, n_layers)

    def forward(self, src):
        # Flip batch and time dim
        src = torch.transpose(src, 0, 1)
        src = self.linear_map(src)
        if self.use_pe:
            src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        # Flip batch and time dim back
        output = torch.transpose(output, 0, 1)
        return output


class RawControlToFeat(nn.Module):
    def __init__(self, network, n_staticfeat, n_timefeat, embedding_dim=50):
        super().__init__()
        self.n_staticfeat = n_staticfeat
        self.n_timefeat = n_timefeat
        self.embedding = nn.Embedding(
            num_embeddings=n_staticfeat,
            embedding_dim=embedding_dim,
        )
        self.network = network

    def forward(self, feat_static, n_timesteps, feat_time=None):
        feat_static = feat_static.type(torch.int64)
        feat_static_embed = self.embedding(
                feat_static.squeeze(dim=-1))[
            :, None, :].repeat(1, n_timesteps, 1)
        if self.n_timefeat > 0:
            assert feat_time is not None, (
                'Time features cannot be None'
                'for n_timefeat > 0.')
            input_to_network = torch.cat(
                [feat_static_embed, feat_time], dim=-1)
        else:
            input_to_network = feat_static_embed
        ctrl_feats = self.network(
            input_to_network
        )
        return ctrl_feats


class ControlToNSTF(nn.Module):
    def __init__(
        self,
        network,
        num_categories,
        d_max,
        d_min=1
    ):
        super().__init__()
        self.network = network
        self.num_categories = num_categories
        assert d_min < d_max
        self.d_max = d_max
        self.d_min = d_min

    def forward(self, ctrl_feats, temperature=1.):
        rho = self.rho(ctrl_feats, temperature=temperature)
        rho = torch.flip(rho, [-1])
        u = 1 - rho / torch.cumsum(rho, -1)
        u = u.flip([-1])
        log_u = torch.log(clamp_probabilities(u))
        return log_u

    def rho(self, ctrl_feats, temperature=1.):
        B, T, _ = ctrl_feats.shape
        u_rho = self.network(ctrl_feats).view(
            B, T, self.num_categories, self.d_max)
        if self.d_min > 1:
            mask = torch.full_like(u_rho, np.log(1e-18))
            log_rho1 = mask[..., :self.d_min - 1]
            log_rho2 = normalize_logprob(
                u_rho[..., self.d_min - 1:],
                axis=-1,
                temperature=temperature)[0]
            log_rho = torch.cat([log_rho1, log_rho2], dim=-1)
            assert log_rho.size() == u_rho.size()
        else:
            log_rho = normalize_logprob(
                u_rho, axis=-1, temperature=temperature)[0]
        rho = clamp_probabilities(torch.exp(log_rho))
        return rho


class InitialDiscreteDistribution(nn.Module):
    def __init__(self, network, num_categories, takes_ctrl=False):
        super().__init__()
        self.network = network
        self.num_categories = num_categories
        self.takes_ctrl = takes_ctrl

    def forward(self, ctrl_feats0):
        B, _ = ctrl_feats0.shape
        if not self.takes_ctrl:
            ctrl_feats0 = torch.ones(B, 1, device=ctrl_feats0.device)
        h = self.network(ctrl_feats0)
        return normalize_logprob(h, axis=-1)[0]


class InitialContinuousDistribution(nn.Module):
    def __init__(
        self,
        networks,
        dist_dim,
        num_categories,
        use_tied_cov=False,
        use_trainable_cov=False,
        sigma=None,
        takes_ctrl=False,
        max_scale=1.,
        scale_nonlinearity='softplus'
    ):
        super().__init__()
        assert len(networks) == num_categories,\
            'The number of networks != num_categories!'
        self.x0_networks = nn.ModuleList(networks)
        self.K = num_categories
        self.use_tied_cov = use_tied_cov
        self.use_trainable_cov = use_trainable_cov
        self.max_scale = max_scale
        self.scale_nonlinearity = scale_nonlinearity
        if not self.use_trainable_cov:
            assert sigma is not None,\
                "sigma cannot be None for non-trainable covariance!"
            self.sigma = sigma
        if self.use_trainable_cov and self.use_tied_cov:
            self.usigma = nn.Parameter(
                inverse_softplus(
                    torch.full(
                        [num_categories, dist_dim],
                        sigma if sigma is not None else 1e-1)))
        self.dist_dim = dist_dim
        self.takes_ctrl = takes_ctrl

    def forward(self, ctrl_feats0):
        B, _ = ctrl_feats0.shape
        if not self.takes_ctrl:
            ctrl_feats0 = torch.ones(B, 1, device=ctrl_feats0.device)
        args_tensor = torch.stack(
            [net(ctrl_feats0) for net in self.x0_networks])
        args_tensor = args_tensor.permute(1, 0, 2)
        mean_tensor = args_tensor[..., :self.dist_dim]
        if self.use_trainable_cov:
            if self.use_tied_cov:
                if self.scale_nonlinearity == 'sigmoid':
                    scale_tensor = self.max_scale * torch.sigmoid(self.usigma)
                else:
                    scale_tensor = F.softplus(self.usigma)
                out_dist = td.normal.Normal(
                    mean_tensor, scale_tensor + SCALE_OFFSET)
                out_dist = td.independent.Independent(out_dist, 1)
            else:
                if self.scale_nonlinearity == 'sigmoid':
                    scale_tensor = self.max_scale * torch.sigmoid(
                        args_tensor[..., self.dist_dim:])
                else:
                    scale_tensor = F.softplus(args_tensor[..., self.dist_dim:])
                out_dist = td.normal.Normal(
                    mean_tensor, scale_tensor + SCALE_OFFSET)
                out_dist = td.independent.Independent(out_dist, 1)
        else:
            out_dist = td.normal.Normal(mean_tensor, self.sigma)
            out_dist = td.independent.Independent(out_dist, 1)
        return out_dist


class ContinuousStateTransition(nn.Module):
    def __init__(
        self,
        transition_networks,
        dist_dim,
        num_categories,
        use_tied_cov=False,
        use_trainable_cov=False,
        sigma=None,
        takes_ctrl=False,
        max_scale=1.,
        scale_nonlinearity='softplus'
    ):
        """A torch module that models p(x[t] | x[t-1], z[t]).

        Args:
            transition_networks (List[nn.Module]):
                List of torch modules of length num_categories.
                The k-th element of the list is a neural network that
                outputs the parameters of the distribution
                p(x[t] | x[t-1], z[t] = k).
            dist_dim (int):
                The dimension of the random variables x[t].
            num_categories (int):
                The number of discrete states.
            use_tied_cov (bool, optional):
                Whether to use a tied covariance matrix.
                Defaults to False.
            use_trainable_cov (bool, optional):
                True if the covariance matrix is to be learned.
                Defaults to True. If False, the covariance matrix is set to I.
            takes_ctrl (bool, optional):
                True if the dataset has control inputs.
                Defaults to False.
            max_scale (float, optional):
                Maximum scale when using sigmoid non-linearity.
            scale_nonlinearity (str, optional):
                Which non-linearity to use for scale -- sigmoid or softplus.
                Defaults to softplus.
        """
        super().__init__()

        assert len(transition_networks) == num_categories,\
            'The number of transition networks != num_categories!'
        self.x_trans_networks = nn.ModuleList(transition_networks)
        self.K = num_categories
        self.use_tied_cov = use_tied_cov
        self.use_trainable_cov = use_trainable_cov
        self.max_scale = max_scale
        self.scale_nonlinearity = scale_nonlinearity
        if not self.use_trainable_cov:
            assert sigma is not None,\
                "sigma cannot be None for non-trainable covariance!"
            self.sigma = sigma
        if self.use_trainable_cov and self.use_tied_cov:
            self.usigma = nn.Parameter(
                inverse_softplus(
                    torch.full([num_categories, dist_dim], 0.1)))
        self.dist_dim = dist_dim
        self.takes_ctrl = takes_ctrl

    def forward(self, x, ctrl_feats):
        """The forward pass.

        Args:
            x (torch.Tensor):
                Pseudo-observations x[1:T] sampled from the variational
                distribution q(x[1:T] | y[1:T]).
                Expected shape: [batch, time, x_dim]

        Returns:
            out_dist:
                The Gaussian distributions p(x[t] | x[t-1], z[t]).
        """
        B, T, dist_dim = x.shape
        assert self.dist_dim == dist_dim,\
            'The distribution dimensions do not match!'
        if self.takes_ctrl:
            assert ctrl_feats is not None, (
                'ctrl_feats cannot be None when self.takes_ctrl = True!')
            # Concat observations and controls on feature dimension
            x = torch.cat([x, ctrl_feats], dim=-1)
        args_tensor = torch.stack(
            [net(x) for net in self.x_trans_networks])
        args_tensor = args_tensor.permute(1, 2, 0, 3)
        mean_tensor = args_tensor[..., :dist_dim]
        if self.use_trainable_cov:
            if self.use_tied_cov:
                if self.scale_nonlinearity == 'sigmoid':
                    scale_tensor = self.max_scale * torch.sigmoid(self.usigma)
                else:
                    scale_tensor = F.softplus(self.usigma)
                out_dist = td.normal.Normal(
                    mean_tensor, scale_tensor + SCALE_OFFSET)
                out_dist = td.independent.Independent(out_dist, 1)
            else:
                if self.scale_nonlinearity == 'sigmoid':
                    scale_tensor = self.max_scale * torch.sigmoid(
                        args_tensor[..., dist_dim:])
                else:
                    scale_tensor = F.softplus(args_tensor[..., dist_dim:])
                out_dist = td.normal.Normal(
                    mean_tensor, scale_tensor + SCALE_OFFSET)
                out_dist = td.independent.Independent(out_dist, 1)
        else:
            out_dist = td.normal.Normal(mean_tensor, self.sigma)
            out_dist = td.independent.Independent(out_dist, 1)
        return out_dist


class DiscreteStateTransition(nn.Module):
    def __init__(
        self,
        transition_network,
        num_categories,
        takes_x=True,
        takes_y=False,
        takes_ctrl=False
    ):
        """A torch module that models p(z[t] | z[t-1], y[t-1]).

        Args:
            transition_network (nn.Module):
                A torch module that outputs the parameters of the
                distribution p(z[t] | z[t-1], y[t-1]).
            num_categories (int):
                The number of discrete states.
            takes_x (bool, optional):
                Whether there is recurrent connection from state to switch.
            takes_y (bool, optional):
                Whether there is recurrent connection from obs to switch.
            takes_ctrl (bool, optional):
                True if the dataset has control inputs.
        """
        super().__init__()
        self.network = transition_network
        self.K = num_categories
        self.takes_ctrl = takes_ctrl
        self.takes_x = takes_x
        self.takes_y = takes_y

    def forward(self, y, x, ctrl_feats=None):
        """The forward pass.

        Args:
            y (torch.Tensor):
                The observations.
                Expected shape: [batch, time, obs_dim]

        Returns:
            transition_tensor:
                The unnormalized transition matrix for each timestep.
                Output shape: [batch, time, num_categories, num_categories]
        """
        B, T = y.shape[:2]
        inputs_to_net = []
        if self.takes_y:
            inputs_to_net += [y]
        if self.takes_x:
            inputs_to_net += [x]
        if self.takes_ctrl:
            assert ctrl_feats is not None, (
                'ctrl_feats cannot be None when self.takes_ctrl = True!')
            inputs_to_net += [ctrl_feats]
        if len(inputs_to_net) == 0:
            # No recurrence
            dummy_inputs = torch.ones(B, T, 1, device=y.device)
            inputs_to_net += [dummy_inputs]
        # Concat on feature dimension
        inputs_to_net = torch.cat(inputs_to_net, dim=-1)
        transition_tensor = self.network(
            inputs_to_net).view([B, T, self.K, self.K])
        return transition_tensor


class GaussianDistributionOutput(nn.Module):
    def __init__(
        self,
        network,
        dist_dim,
        use_tied_cov=False,
        use_trainable_cov=True,
        sigma=None,
        max_scale=1.,
        scale_nonlinearity='softplus'
    ):
        """A Gaussian distribution on top of a neural network.

        Args:
            network (nn.Module):
                A torch module that outputs the parameters of the
                Gaussian distribution.
            dist_dim ([type]):
                The dimension of the Gaussian distribution.
            use_tied_cov (bool, optional):
                Whether to use a tied covariance matrix.
                Defaults to False.
            use_trainable_cov (bool, optional):
                True if the covariance matrix is to be learned.
                Defaults to True. If False, the covariance matrix is set to I.
            sigma (float, optional):
                Initial value of scale.
            max_scale (float, optional):
                Maximum scale when using sigmoid non-linearity.
            scale_nonlinearity (str, optional):
                Which non-linearity to use for scale -- sigmoid or softplus.
                Defaults to softplus.
        """
        super().__init__()
        self.dist_dim = dist_dim
        self.network = network
        self.use_tied_cov = use_tied_cov
        self.use_trainable_cov = use_trainable_cov
        self.max_scale = max_scale
        self.scale_nonlinearity = scale_nonlinearity
        if not self.use_trainable_cov:
            assert sigma is not None,\
                "sigma cannot be None for non-trainable covariance!"
            self.sigma = sigma
        if self.use_trainable_cov and self.use_tied_cov:
            self.usigma = nn.Parameter(
                inverse_softplus(
                    torch.full(
                        [1, dist_dim],
                        sigma if sigma is not None else 1e-1)
                )
            )

    def forward(self, tensor):
        """The forward pass.

        Args:
            tensor (torch.Tensor):
                The input tensor.

        Returns:
            out_dist:
                The Gaussian distribution with parameters obtained by passing
                the input tensor through self.network.
        """
        args_tensor = self.network(tensor)
        mean_tensor = args_tensor[..., :self.dist_dim]
        if self.use_trainable_cov:
            if self.use_tied_cov:
                if self.scale_nonlinearity == 'sigmoid':
                    scale_tensor = self.max_scale * torch.sigmoid(self.usigma)
                else:
                    scale_tensor = F.softplus(self.usigma)
                out_dist = td.normal.Normal(
                    mean_tensor, scale_tensor + SCALE_OFFSET)
                out_dist = td.independent.Independent(out_dist, 1)
            else:
                if self.scale_nonlinearity == 'sigmoid':
                    scale_tensor = self.max_scale * torch.sigmoid(
                        args_tensor[..., self.dist_dim:])
                else:
                    scale_tensor = F.softplus(
                        args_tensor[..., self.dist_dim:])

                out_dist = td.normal.Normal(
                    mean_tensor, scale_tensor + SCALE_OFFSET)
                out_dist = td.independent.Independent(out_dist, 1)
        else:
            out_dist = td.normal.Normal(mean_tensor, self.sigma)
            out_dist = td.independent.Independent(out_dist, 1)
        return out_dist


class RNNInferenceNetwork(nn.Module):
    def __init__(
        self,
        posterior_rnn,
        posterior_dist,
        x_dim,
        embedding_network=None,
        takes_ctrl=False
    ):
        """A torch module that models, q(x[1:T] | y[1:T]), the variational
        distribution of the continuous state.

        Args:
            posterior_rnn (nn.Module):
                The causal rnn with the following recurrence:
                    r[t] = posterior_rnn([h[t], x[t-1]], r[t-1])
                where r is hidden state of the rnn, x is the continuous
                latent variable, and h is an embedding of the observations y
                which is output by the embedding_network.
            posterior_dist (nn.Module):
                A torch module that models q(x[t] | x[1:t-1], y[1:T]) given the
                hidden state r[t] of the posterior_rnn.
            x_dim (int):
                The dimension of the random variables x[t].
            embedding_network (nn.Module, optional):
                A neural network that outputs the embedding h[1:T]
                of the observations y[1:T]. Defaults to None.
            takes_ctrl (bool, optional):
                True if the dataset has control inputs.
        """
        super().__init__()
        self.x_dim = x_dim
        self.posterior_rnn = posterior_rnn
        self.posterior_dist = posterior_dist
        if embedding_network is None:
            embedding_network = lambda x: x  # noqa: E731
        self.embedding_network = embedding_network
        self.takes_ctrl = takes_ctrl

    def forward(self, y, ctrl_feats=None, num_samples=1, deterministic=False):
        """The forward pass.

        Args:
            y (torch.Tensor):
                The observations.
                Expected shape: [batch, time, obs_dim]
            num_samples (int):
                Number of samples for the Importance Weighted (IW) bound.

        Returns:
            x_samples:
                Pseudo-observations x[1:T] sampled from the variational
                distribution q(x[1:T] | y[1:T]).
                Output shape: [batch, time, x_dim]
            entropies:
                The entropy of the variational posterior q(x[1:T] | y[1:T]).
                Each element on the time axis represents the entropy of
                the distribution q(x[t] | x[1:t-1], y[1:T]).
                Output shape: [batch, time]
            log_probs:
                The log probability q(x[1:T] | y[1:T]).
                Each element on the time axis represents the log probability
                q(x[t] | x[1:t-1], y[1:T]).
                Output shape: [batch, time]
        """
        B, T = y.shape[:2]
        latent_dim = self.x_dim
        y = self.embedding_network(y)
        if self.takes_ctrl:
            assert ctrl_feats is not None, (
                'ctrl_feats cannot be None when self.takes_ctrl = True!')
            # Concat observations and controls on feature dimension
            y = torch.cat([y, ctrl_feats], dim=-1)
        if self.posterior_rnn is not None:
            #  Initialize the hidden state or the RNN and the latent sample
            h0 = torch.zeros(B * num_samples, self.posterior_rnn.hidden_size,
                             device=y.device)
            l0 = torch.zeros(B * num_samples, latent_dim,
                             device=y.device)
            hh, ll = h0, l0
        x_samples = []
        entropies = []
        log_probs = []
        for t in range(T):
            yt = y[:, t, :]
            #  Repeat y samples for braodcast
            yt_tiled = yt.repeat(num_samples, 1)

            if self.posterior_rnn is not None:
                #  Concatenate yt with x[t-1]
                rnn_in = torch.cat([yt_tiled, ll], dim=-1)
                #  Feed into the RNN cell and get the hidden state
                hh = self.posterior_rnn(rnn_in, hh)
            else:
                hh = yt_tiled
            #  Construct the distribution q(x[t] | x[1:t-1], y[1:T])
            dist = self.posterior_dist(hh)
            #  Sample from ll ~ q(x[t] | x[1:t-1], y[1:T])
            if deterministic:
                ll = dist.mean
            else:
                ll = dist.rsample()
            x_samples.append(ll)
            #  Compute the entropy of q(x[t] | x[1:t-1], y[1:T])
            entropies.append(dist.entropy())
            #  Compute the log_prob of ll under q(x[t] | x[1:t-1], y[1:T])
            log_probs.append(dist.log_prob(ll))

        x_samples = torch.stack(x_samples)
        # T x (B * num_samples) x latent_dim
        x_samples = x_samples.permute(1, 0, 2)\
            .view(num_samples, B, T, latent_dim)
        entropies = torch.stack(entropies)  # T x (B * num_samples)
        entropies = entropies.permute(1, 0).view(num_samples, B, T)
        log_probs = torch.stack(log_probs)  # T x (B * num_samples)
        log_probs = log_probs.permute(1, 0).view(num_samples, B, T)
        return x_samples, entropies, log_probs
