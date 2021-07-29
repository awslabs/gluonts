import torch
import numpy as np
import random
import pickle

# import the experiment setting

from configs.config_random_15_new import opt

# load the data
from torch.utils.data import DataLoader
from dataset_utils.dataset import ToyDataset, SeqToyDataset

# actually the config doesn't change much
# from configs.config_random_60_new import opt

np.random.seed(opt.seed)
random.seed(opt.seed)
torch.manual_seed(opt.seed)

if opt.model == "DANN":
    from model.model import DANN as Model
elif opt.model == "GDA":
    from model.model import GDA as Model
elif opt.model == "CDANN":
    from model.model import CDANN as Model
    opt.cond_disc = True
elif opt.model == "ADDA":
    from model.model import ADDA as Model
elif opt.model == "MDD":
    from model.model import MDD as Model 
model = Model(opt).to(opt.device)  # .double()

data_source = opt.dataset

with open(data_source, "rb") as data_file:
    data_pkl = pickle.load(data_file)
print(f"Data: {data_pkl['data'].shape}\nLabel: {data_pkl['label'].shape}")

# build dataset
opt.A = data_pkl['A']

data = data_pkl['data']
data_mean = data.mean(0, keepdims=True)
data_std = data.std(0, keepdims=True)
data_pkl['data'] = (data - data_mean) / data_std  # normalize the raw data
datasets = [ToyDataset(data_pkl, i, opt) for i in range(opt.num_domain)]  # sub dataset for each domain


dataset = SeqToyDataset(datasets, size=len(datasets[0]))  # mix sub dataset to a large one
dataloader = DataLoader(
    dataset=dataset,
    shuffle=True,
    batch_size=opt.batch_size
)


# train
for epoch in range(opt.num_epoch):
    model.learn(epoch, dataloader)
    if (epoch + 1) % opt.save_interval == 0 or (epoch + 1) == opt.num_epoch:
        model.save()
    if (epoch + 1) % opt.test_interval == 0 or (epoch + 1) == opt.num_epoch:    
        model.test(epoch, dataloader)
