import torch
from torch_extensions.distributions.stable_relaxed_categorical import \
    StableRelaxedOneHotCategorical
from models_new_will_replace.sgls_rbpf import SwitchingGaussianLinearSystemBaseRBSMC


class CategoricalSwitchingGaussianLinearSystemRBSMC(
    SwitchingGaussianLinearSystemBaseRBSMC
):
    def __init__(self, temperature, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # nn.Parameter so it is registered and works with .to(device) etc.
        self._temperature = torch.nn.Parameter(temperature, requires_grad=False)

    @property
    def temperature(self):
        return self._temperature

    @temperature.setter
    def temperature(self, temperature):
        if isinstance(temperature, float):
            temperature = torch.tensor(temperature)
        self._temperature = temperature

    def _make_switch_model_dist(self, *args, **kwargs):
        if self.temperature is None:
            return super()._make_switch_model_dist(*args, **kwargs)
        else:
            return StableRelaxedOneHotCategorical(
                logits=super()._make_switch_model_dist(*args, **kwargs).logits,
                temperature=self.temperature,
            )

    def _make_switch_proposal_dist(self, *args, **kwargs):
        if self.temperature is None:
            return super()._make_switch_proposal_dist(*args, **kwargs)
        else:
            return StableRelaxedOneHotCategorical(
                logits=super()
                ._make_switch_proposal_dist(*args, **kwargs)
                .logits,
                temperature=self.temperature,
            )