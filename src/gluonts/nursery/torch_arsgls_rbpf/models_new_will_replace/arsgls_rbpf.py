from models_new_will_replace.asgls_rbpf \
    import AuxiliarySwitchingGaussianLinearSystemRBSMC
from models_new_will_replace.rsgls_rbpf import RecurrentSwitchingGaussianLinearSystemRBSMC



class AuxiliaryRecurrentSwitchingGaussianLinearSystemRBSMC(
    AuxiliarySwitchingGaussianLinearSystemRBSMC,
    RecurrentSwitchingGaussianLinearSystemRBSMC,
):
    pass