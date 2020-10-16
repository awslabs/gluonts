import math
from copy import deepcopy
from experiments.pymunk.configs import config_kvae_rb_learn


config = deepcopy(config_kvae_rb_learn.config)
config.experiment_name = "kvae_mc_fix"

# MC
config.rao_blackwellized = False
config.reconstruction_weight = 0.3

# FIXED COVS - as in original paper implementation:
config.init_scale_R_diag = math.sqrt(0.08)
config.init_scale_Q_diag = math.sqrt(0.03)
config.requires_grad_R = False
config.requires_grad_Q = False
config.init_scale_C = 0.05  # taken from original KVAE code
config.lr = 7e-3  # as in original paper (which does not use RB)
config.grad_clip_norm = 150
config.LRinv_logdiag_scaling = 1.0
config.LQinv_logdiag_scaling = 1.0
config.A_scaling = 1.0
config.B_scaling = 1.0 # not used though, no controls
config.C_scaling = 1.0
D_scaling = 1.0  # not used though, no controls
LSinv_logdiag_scaling = 1.0  # not used though in KVAE
F_scaling = 1.0  # not used though in KVAE