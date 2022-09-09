from inspect import signature
from typing import Callable, Dict, Any


def matching_arguments(f: Callable, args: Dict[str, Any]) -> Dict[str, Any]:
    return {p: args[p] for p in signature(f).parameters.keys() & args.keys()}
