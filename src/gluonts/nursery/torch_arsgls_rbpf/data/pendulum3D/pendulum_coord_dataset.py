import numpy as np
import torch
from torch.utils.data import Dataset


class PendulumCoordDataset(Dataset):
    def __init__(self, file_path: str, transform: (callable, None) = None,
                 n_timesteps=None):
        assert callable(transform) or transform is None
        self._data, self._data_gt, self._states = self._load_dataset(file_path)
        if n_timesteps is not None:
            for name in self._data.keys():
                self._data[name] = self._data[name][:, :n_timesteps]
            for name in self._states.keys():
                self._states[name] = self._states[name][:, :n_timesteps]
        self.transform = transform

    def __len__(self):
        idx_batch = 0  # Original data stored as batch x time, transformed to time x batch
        sizes = [val.shape[idx_batch] for val in self._data.values()]
        assert all(size == sizes[0] for size in sizes)
        size = sizes[0]
        return size

    def __getitem__(self, idx):
        idx = idx.tolist() if torch.is_tensor(idx) else idx
        sample = {name: data_source[idx] for name, data_source in
                  self._data.items()}
        transformed_sample = self.transform(
            sample) if self.transform is not None else sample

        sample_gt = {name: data_source[idx] for name, data_source in
                     self._data_gt.items()}
        transformed_sample_gt = self.transform(
            sample_gt) if self.transform is not None else sample_gt

        states = {name: data_source[idx] for name, data_source in
                  self._states.items()}
        transformed_states = self.transform(
            states) if self.transform is not None else states

        return {**transformed_states, **transformed_sample_gt,
                **transformed_sample}

    def _load_dataset(self, file_path):
        npzfile = np.load(file_path)
        data = {"y": npzfile['obs'].astype(np.float32)}
        data_gt = {"y_gt": npzfile['obs_gt'].astype(np.float32)}
        assert data['y'].ndim == 3
        assert data_gt['y_gt'].ndim == 3

        if 'state' in npzfile:  # all except Pong have state.
            position = npzfile['state'].astype(np.float32)[:, :, 0:1]
            velocity = npzfile['state'].astype(np.float32)[:, :, 1:2]
            states = {"position": position, "velocity": velocity}
        else:
            states = None
        return data, data_gt, states
