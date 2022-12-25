from typing import Tuple
from dataclasses import dataclass
from experiments.base_config import BaseConfig


@dataclass()
class PymunkConfig(BaseConfig):
    dims_img: Tuple[int, int, int]
    dims_filter: Tuple[int]
    kernel_sizes: Tuple[int]
    strides: Tuple[int]
    paddings: Tuple[int]
    upscale_factor: int
    prediction_length: int
    lr_decay_steps: int
    lr_decay_rate: float
    grad_clip_norm: float
    batch_size_eval: int
    weight_decay: float
    num_samples_eval: int
    n_epochs_no_resampling: int
    n_epochs_freeze_gls_params: int


def make_model(config):
    from experiments.pymunk.configs.config_kvae import _make_kvae
    from experiments.pymunk.configs.config_arsgls import _make_asgls
    from experiments.pymunk.pymunk_model import PymunkModel

    if config.experiment_name.startswith("kvae"):
        ssm = _make_kvae(config=config)
    elif config.experiment_name == "arsgls":
        ssm = _make_asgls(config=config)
    else:
        raise Exception("bad experiment name")

    model = PymunkModel(
        config=config,
        ssm=ssm,
        dataset_name=config.dataset_name,
        lr=config.lr,
        lr_decay_rate=config.lr_decay_rate,
        lr_decay_steps=config.lr_decay_steps,
        weight_decay=config.weight_decay,
        n_epochs=config.n_epochs,
        batch_sizes={
            "train": config.dims.batch,
            "val": config.batch_size_eval,
            "test": config.batch_size_eval,
        },
        past_length=config.dims.timesteps,
        n_particle_train=config.dims.particle,
        n_particle_eval=config.num_samples_eval,
        prediction_length=config.prediction_length,
        n_epochs_no_resampling=config.n_epochs_no_resampling,
        num_batches_per_epoch=None,
        log_param_norms=False,
        n_epochs_freeze_gls_params=config.n_epochs_freeze_gls_params,
    )
    return model

