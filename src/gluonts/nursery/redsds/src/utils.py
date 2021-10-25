import os
import yaml
import torch
import numpy as np
from datetime import datetime


def normalize_logprob(tensor, axis=-1, temperature=1.):
    tensor = tensor / temperature
    normalizer = torch.logsumexp(tensor, dim=axis, keepdim=True)
    tensor = tensor - normalizer
    normalizer = normalizer.squeeze(axis)
    return tensor, normalizer


def inverse_softplus(tensor: torch.Tensor):
    return tensor + torch.log(1 - torch.exp(-tensor))


def get_precision(x):
    if isinstance(x, float):
        eps = np.finfo(np.array(x).dtype).eps
    elif isinstance(x, (torch.Tensor,)):
        eps = torch.finfo(x.dtype).eps
    elif isinstance(x, (np.ndarray, np.generic)):
        eps = np.finfo(x.dtype).eps
    else:
        raise ValueError("unknown dtype of {}".format(x))
    return eps


def clamp_probabilities(tensor):
    eps = get_precision(tensor)
    clamped_tensor = tensor.clamp(eps, 1 - eps)
    return clamped_tensor


def _learning_rate_warmup(global_step,
                          warmup_end_lr,
                          warmup_start_lr,
                          warmup_steps):
    """Linear learning rate warm-up."""
    p = global_step / warmup_steps
    diff = warmup_end_lr - warmup_start_lr
    return warmup_start_lr + diff * p


def get_learning_rate(global_step, config):
    """Construct Learning Rate Schedule."""
    if config['flat_learning_rate']:
        lr_schedule = config['learning_rate']
    else:
        if global_step < config['warmup_steps_lr']:
            lr_schedule = _learning_rate_warmup(
                global_step,
                config['learning_rate'],
                config['warmup_start_lr'],
                config['warmup_steps_lr'])
        else:
            decay_steps = config['num_steps'] - config['warmup_steps_lr']
            decay_steps = min(
                config.get('lr_decay_steps', decay_steps),
                decay_steps)
            step = min(global_step, decay_steps)
            cosine_decay = 0.5 * (1 + np.cos(np.pi * step / decay_steps))
            alpha = 1.e-2
            decayed = (1 - alpha) * cosine_decay + alpha
            lr_schedule = config['learning_rate'] * decayed
    return lr_schedule


def _schedule_exponential_decay(
                        global_step,
                        decay_steps,
                        kickin_steps,
                        init_val,
                        annealing_rate,
                        min_val=1e-10):
    """Flat and exponential decay schedule."""
    global_step = global_step
    decay_schedule = (
        init_val * annealing_rate **
        (max(global_step - kickin_steps, 0.) / decay_steps))
    temp_schedule = init_val if global_step < kickin_steps\
        else max(decay_schedule, min_val)
    return temp_schedule


def get_temperature(step, config, prefix=''):
    """Construct Temperature Annealing Schedule."""
    if config[prefix + 't_annealing']:
        temperature_schedule = _schedule_exponential_decay(
            step,
            decay_steps=config[prefix + 't_annealing_steps'],
            kickin_steps=config[prefix + 't_annealing_kickin_steps'],
            init_val=config[prefix + 't_init'],
            annealing_rate=config[prefix + 't_annealing_rate'],
            min_val=config[prefix + 't_min'])
    else:
        temperature_schedule = config[prefix + 't_min']
    return temperature_schedule


def get_cross_entropy_coef(step, config):
    """Construct Cross Entropy Coefficient Schedule."""
    if config['xent_annealing']:
        cross_entropy_schedule = _schedule_exponential_decay(
            step,
            decay_steps=config['xent_steps'],
            kickin_steps=config['xent_kickin_steps'],
            init_val=config['xent_init'],
            annealing_rate=config['xent_rate'])
    else:
        cross_entropy_schedule = 0.
    return cross_entropy_schedule


def get_config_and_setup_dirs(filename='config.yaml'):
    with open(filename, 'r') as fp:
        config = yaml.full_load(fp)

    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")

    config['log_dir'] = config['log_dir'].format(timestamp=timestamp)
    config['model_dir'] = config['model_dir'].format(timestamp=timestamp)
    os.makedirs(config['log_dir'])
    os.makedirs(config['model_dir'])

    return config
