import contextlib
import hashlib
import numpy as np
import torch


@contextlib.contextmanager
def local_seed(seed):
    state_np = np.random.get_state()
    state_torch = torch.random.get_rng_state()
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state_np)
        torch.random.set_rng_state(state_torch)


def seed_from(*args):
    """
    hashes any arguments to create a unique seed. This is useful for reproducibility,
    e.g. when generating same data for different models (which increment the seed differently).
    -------------
    Example usage:
    with local_seed(seed_from("train", 10)):
        x = torch.randn(2)
        y = np.random.randn(2)
    z = torch.randn(2)  # here we have re-stored the "global" seed.
    """
    m = hashlib.sha256()
    for arg in args:
        m.update(str(arg).encode())
    h = int.from_bytes(m.digest(), "big")
    seed = h % (2 ** 32 - 1)
    return seed
