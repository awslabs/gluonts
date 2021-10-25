import torch
import numpy as np

from .utils import normalize_logprob, clamp_probabilities, get_precision


def forward_pass_hmm(log_a, log_b, logprob_z0):
    B, T, K, _ = log_a.shape
    device = log_a.device
    forward_probs = torch.zeros(B, T, K, device=device)
    normalizers = torch.zeros(B, T, device=device)

    forward_probs[:, 0, :], normalizers[:, 0] = normalize_logprob(
        logprob_z0[:, :] + log_b[:, 0, :], axis=-1)

    for t in range(1, T):
        log_b_t = log_b[:, t, :]
        log_a_t = log_a[:, t, :, :]
        prev_prob = forward_probs[:, t-1, :]
        alpha = torch.logsumexp(
            log_b_t[:, :, None] + log_a_t + prev_prob[:, None, :], dim=-1)
        forward_probs[:, t, :], normalizers[:, t] = normalize_logprob(
            alpha, axis=-1)
    return forward_probs, normalizers


def backward_pass_hmm(log_a, log_b, logprob_z0):
    B, T, K, _ = log_a.shape
    device = log_a.device
    backward_probs = torch.zeros(B, T, K, device=device)

    for t in range(T-2, -1, -1):
        log_b_t = log_b[:, t+1, :]
        log_a_t = log_a[:, t+1, :, :]
        next_prob = backward_probs[:, t+1, :]
        beta = torch.logsumexp(
            log_b_t[:, :, None] + log_a_t + next_prob[:, :, None], dim=-2)
        backward_probs[:, t, :] = beta
    return backward_probs


def forward_backward_hmm(log_a, log_b, logprob_z0):
    B, T, K, _ = log_a.shape
    device = log_a.device
    _ones = torch.ones(B, T, K, device=device)
    _A_summed = torch.exp(log_a).sum(-1)
    if not torch.allclose(
        _ones,
        _A_summed,
        atol=1e-7
    ):
        unique_diff_vals = torch.unique(
            _A_summed[_A_summed != _ones].data).cpu().numpy()
        print(
         'WARNING: Transition probabilities do not sum to 1!\n'
         'Unique differing values:\n'
         f'{unique_diff_vals}')
        if torch.isnan(log_a).any():
            raise RuntimeError('Crashed with NaNs!')
    log_a = torch.transpose(log_a, -2, -1)
    fwd, fwd_normalizers = forward_pass_hmm(log_a, log_b, logprob_z0)
    bwd = backward_pass_hmm(log_a, log_b, logprob_z0)

    m_fwd = fwd[:, 0:-1, None, :]
    m_bwd = bwd[:, 1:, :, None]
    m_a = log_a[:, 1:, :, :]
    m_b = log_b[:, 1:, :, None]

    # posterior score
    log_gamma = fwd + bwd
    log_xi = m_fwd + m_a + m_bwd + m_b

    # normalize the probability matrices
    log_gamma, _ = normalize_logprob(log_gamma, axis=-1)
    log_xi_flat, _ = normalize_logprob(
        log_xi.reshape(B, T-1, K * K),
        axis=-1)
    log_xi = log_xi_flat.view_as(log_xi)

    # padding the matrix to the same shape of inputs
    log_uniform_pad = torch.log(
        torch.ones([B, 1, K, K], device=device) / (K * K))
    log_xi = torch.cat([log_uniform_pad,
                       log_xi], dim=1)
    log_p_XY = fwd_normalizers.sum(-1)
    return fwd, bwd, log_gamma, log_xi, log_p_XY


def forward_pass_hsmm(log_a, log_b, logprob_z0, log_u):
    B, T, K, _ = log_a.shape
    _, _, _, d_max = log_u.shape
    device = log_a.device
    eps = get_precision(log_a)
    forward_probs = torch.zeros(B, T, K, d_max, device=device)
    forward_probs[:, 0, :, 0] = logprob_z0[:, :] + log_b[:, 0, :]
    forward_probs[:, 0, :, 1:] = np.log(eps)
    for t in range(1, T):
        forward_probs[:, t, :, 1:] = log_b[:, t, :, None] +\
             log_u[:, t, :, :-1] + forward_probs[:, t-1, :, :-1]
        tmp = clamp_probabilities(1 - log_u[:, t, :, :].exp()).log()\
            + forward_probs[:, t-1, :, :]
        tmp = torch.logsumexp(tmp, -1)
        forward_probs[:, t, :, 0] = torch.logsumexp(
            log_b[:, t, None, :] + log_a[:, t, :, :] + tmp[:, :, None], dim=-2)
    return forward_probs


def backward_pass_hsmm(log_a, log_b, logprob_z0, log_u):
    B, T, K, _ = log_a.shape
    _, _, _, d_max = log_u.shape
    device = log_a.device
    eps = get_precision(log_a)
    backward_probs = torch.full(
        (B, T, K, d_max), np.log(eps), device=device)
    backward_probs[:, -1, :, :] = 0.
    for t in range(T-2, -1, -1):
        beta = torch.full((B, K, d_max), np.log(eps), device=device)
        beta[:, :, :-1] = backward_probs[:, t+1, :, 1:] +\
            log_b[:, t+1, :, None] + log_u[:, t+1, :, :-1]
        tmp = backward_probs[:, t+1, None, :, 0] + log_b[:, t+1, None, :] +\
            log_a[:, t+1, :, :]
        tmp = torch.logsumexp(tmp, dim=-1)
        tmp2 = tmp[:, :, None]\
            + clamp_probabilities(1 - log_u[:, t+1, :, :].exp()).log()
        backward_probs[:, t, :, :] = torch.logsumexp(
            torch.stack([beta, tmp2]), dim=0)
    return backward_probs


def forward_backward_hsmm(log_a, log_b, logprob_z0, log_u):
    B, T, K, _ = log_a.shape
    _, _, _, d_max = log_u.shape
    assert log_a.shape[:-1] == log_u.shape[:-1],\
        'log_a and log_u shapes should match in all dims except last!'
    device = log_a.device
    _ones = torch.ones(B, T, K, device=device)
    _A_summed = torch.exp(log_a).sum(-1)
    if not torch.allclose(
        _ones,
        _A_summed,
        atol=1e-7
    ):
        unique_diff_vals = torch.unique(
            _A_summed[_A_summed != _ones].data).cpu().numpy()
        print(
         'WARNING: Transition probabilities do not sum to 1!\n'
         'Unique differing values:\n'
         f'{unique_diff_vals}')
        if torch.isnan(log_a).any():
            raise RuntimeError('Crashed with NaNs!')

    fwd = forward_pass_hsmm(log_a, log_b, logprob_z0, log_u)
    bwd = backward_pass_hsmm(log_a, log_b, logprob_z0, log_u)

    # posterior score
    log_gamma = fwd + bwd
    log_gamma_flat, _ = normalize_logprob(
        log_gamma.view(B, T, K * d_max), axis=-1)
    log_gamma = log_gamma_flat.view_as(log_gamma)
    log_p_XY = torch.logsumexp(fwd[:, -1].view(B, K * d_max), dim=-1)
    return fwd, bwd, log_gamma, log_p_XY
