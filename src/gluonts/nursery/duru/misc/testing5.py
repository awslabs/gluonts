import torch
import torch.distributions as D


def gaussian_analytical_kl(mu1, mu2, log_sigma1, log_sigma2):
    """
    KL[p_1(m0,log_sigma0), p_2(mu1, log_sigma1)]
    """
    log_var_ratio = 2 * (log_sigma1 - log_sigma2)
    t1 = (mu1 - mu2) ** 2 / (2 * log_sigma2).exp()
    return 0.5 * (log_var_ratio.exp() + t1 - 1 - log_var_ratio)


# (Batch, Features, Timesteps)
q_loc = torch.randn(10, 5, 20)
q_scale = torch.clip(torch.randn(10, 5, 20), min=0.001)
q = D.Independent(
    D.Normal(loc=q_loc, scale=q_scale), reinterpreted_batch_ndims=2
)

p_loc = torch.randn(10, 5, 20)
p_scale = torch.clip(torch.randn(10, 5, 20), min=0.001)
p = D.Independent(
    D.Normal(loc=p_loc, scale=p_scale), reinterpreted_batch_ndims=2
)

res1 = D.kl.kl_divergence(q, p)
res2 = gaussian_analytical_kl(
    q_loc, p_loc, torch.log(q_scale), torch.log(p_scale)
).sum(dim=(1, 2))

print("res1")
print(res1)
print("res2")
print(res2)
