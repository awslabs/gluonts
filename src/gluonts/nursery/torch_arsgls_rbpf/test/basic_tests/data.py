import os
from torch.utils.data import DataLoader

import consts
from data.pymunk_kvae.pymunk_dataset import PymunkDataset
from data.pendulum3D.pendulum_image_dataset import PendulumImageDataset
from data.pendulum3D.pendulum_coord_dataset import PendulumCoordDataset
from data.transforms import time_first_collate_fn


def _is_data_from_loader_shuffled(dataloader):
    b1 = next(iter(dataloader))
    b2 = next(iter(dataloader))
    data_names = tuple(b1.keys())
    is_shuffled = not all((b1[name] == b2[name]).all() for name in data_names)
    return is_shuffled


_dataset_name_to_cls = {
    consts.Datasets.box: PymunkDataset,
    consts.Datasets.box_gravity: PymunkDataset,
    consts.Datasets.pong: PymunkDataset,
    consts.Datasets.polygon: PymunkDataset,
    consts.Datasets.pendulum_3D_image: PendulumImageDataset,
    consts.Datasets.pendulum_3D_coord: PendulumCoordDataset,
}


def _test_loaders_shuffling(dataset_name, batch_size=32, num_workers=2):
    dataset_cls = _dataset_name_to_cls[dataset_name]
    dataloaders = {
        data_subset_name: DataLoader(
            dataset=dataset_cls(
                file_path=os.path.join(
                    consts.data_dir, dataset_name, f"{data_subset_name}.npz"
                )
            ),
            batch_size=batch_size,
            shuffle=True if data_subset_name == "train" else False,
            num_workers=num_workers,
            collate_fn=time_first_collate_fn,
        )
        for data_subset_name in ["train", "test"]
    }
    assert _is_data_from_loader_shuffled(dataloaders["train"]) is True
    assert _is_data_from_loader_shuffled(dataloaders["test"]) is False


def test_box_loaders():
    _test_loaders_shuffling(dataset_name=consts.Datasets.box)


def test_box_gravity_loaders():
    _test_loaders_shuffling(dataset_name=consts.Datasets.box_gravity)


def test_polygon_loaders():
    _test_loaders_shuffling(dataset_name=consts.Datasets.polygon)


def test_pong_loaders():
    _test_loaders_shuffling(dataset_name=consts.Datasets.pong)


def test_pendulum_3d_coord_loaders():
    _test_loaders_shuffling(dataset_name=consts.Datasets.pendulum_3D_coord)


def test_pendulum_3d_image_loaders():
    _test_loaders_shuffling(dataset_name=consts.Datasets.pendulum_3D_image)
