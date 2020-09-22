from typing import Optional

import torch


def copy_parameters(
    net_source: torch.nn.Module,
    net_dest: torch.nn.Module,
    strict: Optional[bool] = True,
) -> None:
    """
    Copies parameters from one network to another.

    Parameters
    ----------
    net_source
        Input network.
    net_dest
        Output network.
    strict:
        whether to strictly enforce that the keys
        in :attr:`state_dict` match the keys returned by this module's
        :meth:`~torch.nn.Module.state_dict` function. Default: ``True``
    """

    net_dest.load_state_dict(net_source.state_dict(), strict=strict)
