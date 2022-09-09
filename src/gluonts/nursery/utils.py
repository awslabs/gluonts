from inspect import signature
from typing import Callable, Dict, Any


def matching_parameters(f: Callable, hps: Dict[str, Any]) -> Dict[str, Any]:
    return {p: hps[p] for p in signature(f).parameters.keys() & hps.keys()}