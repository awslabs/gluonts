from torch.distributions.transforms import ExpTransform
from torch.distributions import RelaxedOneHotCategorical
from torch.distributions.relaxed_categorical import ExpRelaxedCategorical

from torch_extensions.ops import stable_log


class StableExpTransform(ExpTransform):
    def _inverse(self, y):  # Use stable log. Leave the rest as is.
        return stable_log(y, eps=1e-20)


class StableRelaxedOneHotCategorical(RelaxedOneHotCategorical):
    """
    Pytorch implementation of RelaxedOneHotCategorical is unstable.
    They use torch.exp and torch.log transforms directly for which the latter fails for
    inputs that are numerically 0 due to imprecision.
    """

    def __init__(
        self, temperature, probs=None, logits=None, validate_args=None
    ):
        base_dist = ExpRelaxedCategorical(temperature, probs, logits)
        # Do *not* call constructor of RelaxedOneHotCategorical, as it has hard-coded
        # the unstable ExpTransform. Call its parent class directly.
        super(RelaxedOneHotCategorical, self).__init__(
            base_dist, StableExpTransform(), validate_args=validate_args
        )
