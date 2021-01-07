from collections import defaultdict
from itertools import count
from pydoc import locate

from toolz.dicttoolz import keymap


def asdict(trie):
    if not isinstance(trie, defaultdict):
        return trie
    return {k: asdict(v) for k, v in trie.items()}


class Trie:
    def __init__(self, dct=None):
        trie = lambda: defaultdict(trie)
        self.trie = trie()

        if dct is not None:
            for key, val in dct.items():
                self[key] = val

    def __setitem__(self, path, value):
        trie = self.trie
        for part in path[:-1]:
            trie = trie[part]

        trie[path[-1]] = value

    def asdict(self):
        return asdict(self.trie)


def load_references(parameters: dict) -> dict:
    """Return a copy of `parameters`, where we load objects, where keys end in
    `@`:

    >>> load_references({"@": "operator.add"})
    {'@': <built-in function add>}
    """
    copy = dict(parameters)
    copy.update(
        {
            key: locate(value)
            for key, value in parameters.items()
            if key[-1] == "@"
        }
    )
    return copy


def eval_nested(data):
    if isinstance(data, dict) and "@" in data:
        Type = data.pop("@")
        args = []

        for idx in map(str, count(start=0)):
            if idx not in data:
                break
            args.append(eval_nested(data.pop(idx)))

        kwargs = {key: eval_nested(value) for key, value in data.items()}
        return Type(*args, **kwargs)

    return data


def decode_nested_parameters(parameters: dict) -> dict:
    def split_path(s):
        return tuple(s.split("."))

    path_parameters = keymap(split_path, parameters)

    parameters_with_objects = load_references(path_parameters)

    nested = Trie(parameters_with_objects).asdict()
    return eval_nested(nested)
