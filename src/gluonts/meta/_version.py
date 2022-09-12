# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

"""
This file supports "dynamic" versions using git during development. When the
package is built, the content of this file is replaced with a static version.

This is loosely based on miniver.

There are conditions for this file to work.

This file needs to be part of the source code. To use it, just import
`__version__` from it::

    # __init__.py
    from _version import __version__

The projects `setup.py` is at most `SEARCH_PACKAGE_LEVELS` directory levels up,
since this file needs to know where it is relative to `setup.py`.

When using git, `setup.py` needs to be located at the root of the repository.

Thus, a typical setup looks like this::

    my_package/
        .git
        my_package/
            __init__.py
            _version.py
        setup.py


In `setup.py`, you need to import and use this file like this:

    def get_version_and_cmdclass(version_file):
        with open(version_file) as version_file:
            code = version_file.read()

        globals_ = {"__file__": str(version_file)}
        exec(code, globals_)

        return globals_["__version__"], globals_["cmdclass"]

    version, vercmdclass = get_version_and_cmdclass("<path>/<to>/_version.py")

    mycmdclass = {...}

    setup(
        cmdclass={**mycmdclass, **vercmdclass},
    )
"""

import subprocess
from pathlib import Path


GIT_DESCRIBE = [
    "git",
    "describe",
    # `--long` produces triple `{tag}-{num-commits}-g{hash}`, where
    #     * `tag` is the most recent tag to be found
    #     * `num-commits` indicates the number of commits since the tag,
    #        where a `0` means that the commit itself is tagged
    #     * `hash` is the 8 character commit hash
    "--long",
    # `--tags` enables use of non-annotated tags
    # (comment out if not needed)
    "--tags",
    # `--match` allows to filter for tags
    # we require tag to start with `v` followed by a number
    "--match=v[0-9]*",
]

SEARCH_PACKAGE_LEVELS = 4


def search_for(name: str, where: Path, levels: int = 0):
    while True:
        candidate = where / name
        if candidate.exists():
            return candidate

        if levels == 0 or where == where.parent or where is None:
            raise RuntimeError(f"Can not find {name}.")

        levels -= 1
        where = where.parent


file_name = Path(__file__).name  # this files name
package_root = Path(__file__).resolve().parents[1]  # 2 levels up


def dist_root():
    return search_for(
        "setup.py", Path(__file__).parent, levels=SEARCH_PACKAGE_LEVELS
    ).parent.resolve()


class GitRepo:
    def __init__(self, cwd):
        self.cwd = cwd

    def sh(self, cmd, **kwargs):
        return (
            subprocess.check_output(cmd, cwd=self.cwd, **kwargs)
            .decode()
            .strip()
        )

    def root(self) -> Path:
        return Path(self.sh(["git", "rev-parse", "--show-toplevel"]))

    def clean(self) -> bool:
        """
        Return if any tracked file has uncommited changes.
        """

        proc = subprocess.run(["git", "diff", "--quiet"], cwd=self.cwd)
        return proc.returncode == 0

    def describe(self):
        """
        Return triple `<tag>-<commits>-<hash>`.
        """
        return self.sh(GIT_DESCRIBE, stderr=subprocess.PIPE)

    def head_commit(self):
        """
        Get hash of most recent commit.
        """
        return self.sh(["git", "rev-parse", "--short", "HEAD"])


def get_git_archive_version(fallback):
    # $...$ strings can be overwitten by `git archive` if `.gitattributes`
    # is configured with `.../_version.py export-subst`

    # if commit is tagged, refnames contains `tag: <name>` pairs for each tag
    refnames = "$Format:%D$"

    if not refnames.startswith("$Format"):
        refs = refnames.split(", ")
        tags = [
            ref.split("tag: ")[1] for ref in refs if ref.startswith("tag:")
        ]

        # if there is a tag, we return the first one
        if tags:
            return tags[0]

    commit_hash = "$Format:%h$"

    if not commit_hash.startswith("$Format"):
        return f"{fallback}+g{commit_hash}"

    return None


def get_git_version(fallback):
    def format_version(release, is_dev, labels):
        dev = ".dev0" if is_dev else ""
        labels = "+" + ".".join(labels) if labels else ""
        return "".join([release, dev, labels])

    dist_root_ = dist_root()
    repo = GitRepo(dist_root_)

    try:
        # TODO: Do we really need this check?
        if repo.root() != dist_root_:
            return None
    except (subprocess.CalledProcessError, FileNotFoundError):
        # can fail if git is not installed, or command fails
        return None

    try:
        git_tag = repo.describe()
        release, num_changes, git_hash = git_tag.rsplit("-", 2)

        release = release.lstrip("v")
        # if the current commit is not tagged directly, num_changes is ` > 0`
        # and attach the `.dev` label
        is_dev = num_changes != "0"
        labels = [git_hash] if is_dev else []

    except subprocess.CalledProcessError:
        # If there is no git-tag, we have to fallback.
        release = fallback
        is_dev = False
        labels = ["g" + repo.head_commit()]

    if not repo.clean():
        labels.append("dirty")

    return format_version(release, is_dev, labels)


def get_version(fallback):
    return (
        get_git_archive_version(fallback)
        or get_git_version(fallback)
        or fallback
    )


def cmdclass():
    import setuptools.command.build_py
    import setuptools.command.sdist

    def write_version(target):
        target /= file_name
        if target.exists():
            target.unlink()

        with open(target, "w") as version_file:
            version_file.write(
                f'# created by setup.py\n__version__ = "{__version__}"'
            )

    class build_py(setuptools.command.build_py.build_py):
        def run(self):
            super().run()

            write_version(
                Path(self.build_lib)
                / package_root.name
                / Path(__file__).parent.resolve().relative_to(package_root)
            )

    class sdist(setuptools.command.sdist.sdist):
        def make_release_tree(self, base_dir, files):
            super().make_release_tree(base_dir, files)

            write_version(
                Path(base_dir) / package_root.relative_to(dist_root())
            )

    return {"sdist": sdist, "build_py": build_py}


__version__ = get_version(fallback="0.0.0")
