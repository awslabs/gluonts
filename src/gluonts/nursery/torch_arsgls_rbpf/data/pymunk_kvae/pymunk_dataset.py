import numpy as np
import torch
from torch.utils.data import Dataset


class PymunkDataset(Dataset):
    def __init__(self, file_path: str, transform: (callable, None) = None):
        assert callable(transform) or transform is None
        self._data, self._ground_truth = self._load_dataset(file_path)
        self.transform = transform

    def __len__(self):
        idx_batch = 0  # Original data stored as batch x time, transformed to time x batch
        sizes = [val.shape[idx_batch] for val in self._data.values()]
        assert all(size == sizes[0] for size in sizes)
        size = sizes[0]
        return size

    def __getitem__(self, idx):
        idx = idx.tolist() if torch.is_tensor(idx) else idx
        sample = {
            name: data_source[idx] for name, data_source in self._data.items()
        }
        transformed_sample = (
            self.transform(sample) if self.transform is not None else sample
        )
        return transformed_sample

    def _load_dataset(self, file_path):
        npzfile = np.load(file_path)
        images = npzfile["images"].astype(np.float32)
        # The datasets in KVAE are binarized images
        images = (images > 0).astype("float32")
        assert images.ndim == 4
        images = images.reshape(
            images.shape[0], images.shape[1], images.shape[2] * images.shape[3]
        )
        data = {"y": images}

        if "state" in npzfile:  # all except Pong have state.
            position = npzfile["state"].astype(np.float32)[:, :, :2]
            velocity = npzfile["state"].astype(np.float32)[:, :, 2:]
            # # KVAE normalizes the position. However, it is used only for visualisation anyways.
            # position = position - position.mean(axis=(0, 1))
            ground_truth_state = {"position": position, "velocity": velocity}
        else:
            ground_truth_state = None
        return data, ground_truth_state
