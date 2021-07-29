import numpy as np
from torch.utils.data import Dataset
import pickle


def read_pickle(name):
    with open(name, "rb") as f:
        data = pickle.load(f)
    return data


def write_pickle(data, name):
    with open(name, "wb") as f:
        pickle.dump(data, f)


# import os
# from random import shuffle
# from utils import read_pickle
# from utils import write_pickle


class ToyDataset(Dataset):
    def __init__(self, pkl, domain_id, opt=None):
        idx = pkl["domain"] == domain_id
        self.data = pkl["data"][idx].astype(np.float32)
        self.label = pkl["label"][idx].astype(np.int64)
        self.domain = domain_id

        # if opt.normalize_domain:
        #     print('===> Normalize in every domain')
        #     self.data_m, self.data_s = self.data.mean(0, keepdims=True), self.data.std(0, keepdims=True)
        #     self.data = (self.data - self.data_m) / self.data_s

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx], self.domain

    def __len__(self):
        return len(self.data)


class SeqToyDataset(Dataset):
    def __init__(self, datasets, size=3 * 200):
        self.datasets = datasets
        self.size = size
        print(
            "SeqDataset Size {} Sub Size {}".format(
                size, [len(ds) for ds in datasets]
            )
        )

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        return [ds[i] for ds in self.datasets]
