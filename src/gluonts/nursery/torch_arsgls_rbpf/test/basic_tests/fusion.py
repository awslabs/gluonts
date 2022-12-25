import torch
from torch.distributions import (
    Normal,
    MultivariateNormal,
    Bernoulli,
    Categorical,
    OneHotCategorical,
)

from torch_extensions.fusion import test_fusion_manually


def test_fusion():
    # threshold only 1e-4 for float32 --> verify with float64
    torch.set_default_dtype(torch.float64)
    for distributions in [
        [
            Normal(loc=torch.rand(10), scale=torch.rand(10) + 1),
            Normal(loc=torch.rand(10), scale=torch.rand(10) + 1),
            Normal(loc=torch.rand(10), scale=torch.rand(10) + 1),
        ],
        [
            MultivariateNormal(
                loc=torch.rand(10),
                covariance_matrix=torch.diag(torch.rand(10)),
            ),
            MultivariateNormal(
                loc=torch.rand(10),
                covariance_matrix=torch.diag(torch.rand(10)),
            ),
            MultivariateNormal(
                loc=torch.rand(10),
                covariance_matrix=torch.diag(torch.rand(10)),
            ),
        ],
        [
            Bernoulli(logits=torch.rand(10)),
            Bernoulli(logits=torch.rand(10)),
            Bernoulli(logits=torch.rand(10)),
        ],
        [
            Categorical(logits=torch.rand(10)),
            Categorical(logits=torch.rand(10)),
            Categorical(logits=torch.rand(10)),
        ],
        [
            OneHotCategorical(logits=torch.rand(10)),
            OneHotCategorical(logits=torch.rand(10)),
            OneHotCategorical(logits=torch.rand(10)),
        ],
    ]:
        test_fusion_manually(distributions, threshold=1e-10)
