from inspect import signature
from typing import Callable, Dict, Any


def matching_arguments(f: Callable, args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract valid keyword arguments for a Callable from a dictionary.

    Parameters
    ----------
    f
        Callable for which matching arguments are required.
    args
        A dictionary containing keyword arguments.

    Returns
    -------
    Dict
        A dictionary containing valid keyword arguments.

    """
    return {p: args[p] for p in signature(f).parameters.keys() & args.keys()}
