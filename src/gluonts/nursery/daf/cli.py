from typing import Text, Any
import argparse
import inspect

__all__ = [
    "NestedNamespace",
    "NestedArgumentGroup",
    "NestedArgumentParser",
]


class NestedNamespace(argparse.Namespace):
    DELIMITER = "."

    def __setattr__(self, name: Text, value: Any):
        if self.DELIMITER in name:
            parent, name = name.split(self.DELIMITER, maxsplit=1)
            if len(parent) == 0 or len(name) == 0:
                raise ValueError(f"Invalid attribute name {name}")
            if not hasattr(self, parent):
                super().__setattr__(parent, NestedNamespace())
            setattr(getattr(self, parent), name, value)
        else:
            super().__setattr__(name, value)

    @property
    def __dict__(self):
        raw_dict = super().__dict__
        return {
            key: vars(value) if isinstance(value, NestedNamespace) else value
            for key, value in raw_dict.items()
        }


class NestedArgumentGroup(argparse._ArgumentGroup):
    def add_argument(self, *args, **kwargs):
        group = self.title
        if group:
            dest = kwargs.pop("dest", None)
            if dest:
                kwargs["dest"] = f"{group}.{dest}"
            elif args:
                args = list(args)
                for i, arg in enumerate(args):
                    if arg.startswith(self.prefix_chars * 2):
                        kwargs[
                            "dest"
                        ] = f"{group}.{arg.lstrip(self.prefix_chars)}"
                        break
                    elif not arg.startswith(self.prefix_chars):
                        args[i] = f"{group}.{arg}"
                        break
        super().add_argument(*args, **kwargs)


class NestedArgumentParser(argparse.ArgumentParser):
    def parse_known_args(self, args=None, namespace=None) -> NestedNamespace:
        if namespace is None:
            namespace = NestedNamespace()
        return super().parse_known_args(args=args, namespace=namespace)

    def add_argument_group(self, *args, **kwargs) -> NestedArgumentGroup:
        group = NestedArgumentGroup(self, *args, **kwargs)
        self._action_groups.append(group)
        return group
