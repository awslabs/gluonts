from models.asgls_rbpf \
    import AuxiliarySwitchingGaussianLinearSystemRBSMC
from models.rsgls_rbpf import RecurrentMixin


class AuxiliaryRecurrentSwitchingGaussianLinearSystemRBSMC(
    RecurrentMixin, AuxiliarySwitchingGaussianLinearSystemRBSMC,
):
    pass
