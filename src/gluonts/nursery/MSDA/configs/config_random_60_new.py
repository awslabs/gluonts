from easydict import EasyDict
import numpy as np
import pickle


def read_pickle(name):
    with open(name, 'rb') as f:
        data = pickle.load(f)
    return data


# load/output dir
opt = EasyDict()  # set experiment configs
opt.loadf = "./dump"
opt.outf = "./dump"

# normalize each data domain
# opt.normalize_domain = False

# now it is half circle
opt.num_domain = 60
# the specific source and target domain:
opt.src_domain = np.array([2, 14, 41, 23, 59, 33])  # 60-spiral
opt.num_source = opt.src_domain.shape[0]
opt.num_target = opt.num_domain - opt.num_source
opt.test_on_all_dmn = True


print("src domain: {}".format(opt.src_domain))

# opt.model = "DANN"
# opt.model = "CDANN"
# opt.model = "ADDA"
# opt.model = 'MDD'
opt.model = 'GDA'
opt.cond_disc = False  # whether use conditional discriminator or not (for CDANN)
print("model: {}".format(opt.model))
opt.use_visdom = False
opt.visdom_port = 2000

# we do not prepare the pretrain g encode for random-60 dataset
opt.use_g_encode = False
# if opt.use_g_encode:
#     opt.g_encode = read_pickle("derive_g_encode/g_encode.pkl")

opt.device = "cuda"
opt.seed = 233  # 1# 101 # 1 # 233 # 1

opt.lambda_gan = 0.5  # 0.5 # 0.3125 # 0.5 # 0.5

# for MDD use only
opt.lambda_src = 0.5
opt.lambda_tgt = 0.5

opt.num_epoch = 500
opt.batch_size = 10
opt.lr_d = 1e-5  # 3e-5 # 1e-4 # 2.9 * 1e-5 #3e-5  # 1e-4
opt.lr_e = 1e-5  # 3e-5 # 1e-4 # 2.9 * 1e-5
opt.lr_g = 3e-4
opt.gamma = 100
opt.beta1 = 0.9
opt.weight_decay = 5e-4
opt.wgan = False  # do not use wgan to train
opt.no_bn = True  # do not use batch normalization # True

# model size configs, used for D, E, F
opt.nx = 2  # dimension of the input data
opt.nt = 2  # dimension of the vertex embedding
opt.nh = 512  # dimension of hidden # 512
opt.nc = 2  # number of label class
opt.nd_out = 2  # dimension of D's output

# sample how many vertices for training D
opt.sample_v = 30

# # sample how many vertices for training G
opt.sample_v_g = 60

opt.test_interval = 20
opt.save_interval = 100
# drop out rate
opt.p = 0.2
opt.shuffle = True

# dataset
opt.dataset = 'data/toy_d60_spiral.pkl'
