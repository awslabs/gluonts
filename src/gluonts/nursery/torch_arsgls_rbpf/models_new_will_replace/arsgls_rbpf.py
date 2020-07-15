from models_new_will_replace.asgls_rbpf \
    import AuxiliarySwitchingGaussianLinearSystemRBSMC
from models_new_will_replace.rsgls_rbpf import RecurrentMixin


class AuxiliaryRecurrentSwitchingGaussianLinearSystemRBSMC(
    RecurrentMixin, AuxiliarySwitchingGaussianLinearSystemRBSMC,
):
    pass
