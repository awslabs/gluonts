import torch


def normalize_log_weights(log_weights: torch.Tensor, dim: int = -2):
    log_norm_weights = log_weights - torch.logsumexp(
        log_weights, dim=dim, keepdim=True
    )
    return log_norm_weights
