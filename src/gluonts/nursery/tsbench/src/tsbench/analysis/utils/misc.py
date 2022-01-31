from collections import defaultdict
from typing import Dict, List
from tsbench.config import ModelConfig


def union_dicts(
    dicts: List[Dict[str, ModelConfig]]
) -> Dict[str, List[ModelConfig]]:
    """
    Merges the dicts by aggregating model configurations with the same key into a list.
    """
    result = defaultdict(list)
    for item in dicts:
        for k, v in item.items():
            result[k].append(v)
    return result
