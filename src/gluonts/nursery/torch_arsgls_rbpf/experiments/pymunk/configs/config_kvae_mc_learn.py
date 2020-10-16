from copy import deepcopy
from experiments.pymunk.configs import config_kvae_rb_learn


config = deepcopy(config_kvae_rb_learn.config)
config.experiment_name = "kvae_mc_learn"

# MC
config.rao_blackwellized = False
config.reconstruction_weight = 0.3

# LEARN - same
