Development Setup
=================

.. highlight:: bash

GluonTS requires Python 3.6 or higher to run. This setup guide assumes that the
``python`` command references a Python 3.6 version or higher. We recommend
using pyenv_ for managing Python versions.

Upon checking out this package, please run the following::

    ./dev_setup.sh
    pip install -e .[dev]

    # if you use zsh you might need to escape `[` and `]`
    pip install -e ".[dev]"

This will install all required packages with pip and setup a Git hook that does
automated type and style checks when you try to create a new Git commit.

When you create commits on a |WIP| branch, you can disable these checks for a
with the ``--no-verify`` Git commit option.

.. _pyenv: https://github.com/pyenv/pyenv

Build Instructions
------------------

To run the project tests::

    pytest
    # or
    python setup.py tests

To build the project documentation::

    python setup.py docs

This will put the documentation in ``docs/_build/html``, where you can inspect
it by opening ``index.html``.

You can also run the code quality checks manually using ``setup.py``::

    python setup.py type_check   # for Mypy type checks
    python setup.py style_check  # for Black code style checks

Note that the above commands are executed automatically as part of the
``build`` command. Developers that want to merge their code changes in the
project mainline are therefore advised to ensure that their commits do not
violate the above commands. If you have configured your developer environment
using ``./dev_setup.sh`` and are not relying on the ``--no-verify`` option,
this should already be asserted when you create your commits.


Writing Type-Safe Code
----------------------

The codebase makes extensive use of `type hints`_. The benefits of using types
are twofold. On the one hand, typing the arguments and the return type of
methods and functions provides additional layer of meta-information and
improves the readability of the code. On the other, with the help of an
external type-checker such as `mypy`_ we can statically catch a number of
corner cases that can possibly cause bugs in production.

As explained above, in order to implement the latter, the ``type_check``
command is executed by the build toolchain. However, note that due to the large
number of type errors currently reported by the ``mypy`` check, at the moment
not all packages are type-checked. Until we finish the migration of existing
packages, developers are advised to implement the following guidelines in order
to ensure that code added to the repository is type-safe and type-checked.

1. Add an empty file called ``.typesafe`` to each new package. Make sure that
   the marker files are added to test packages as well. You may omit this file
   if an ancestor package already is marked as ``.typesafe``.

2. Make sure that you provide proper type annotations for the parameters and
   the return type of every new method and function. This also applies to
   `void` method and functions as well as ``__init__`` methods, which should be
   marked with return type ``None``.

If you adhere to the above guidelines, you should be able to run
``python setup.py style_check`` and catch type errors early directly on your
|WIP| branch.

.. _type hints: https://docs.python.org/3.6/library/typing.html
.. _mypy: https://mypy.readthedocs.io/en/latest/

Editing the documentation
-------------------------

GluonTS documentation follows the `NumPy docstring format`_.

If you are editing docstrings in source code, you can preview them with the
following commands::

    make -C docs html                # generate the docs
    open docs/_build/html/index.html # open the generated docs in a browser

Ensure that there are no syntax errors and warnings before committing a PR.

If you are directly editing ``*.rst`` files within the ``docs`` folder, you
can use a ``sphinx-autobuild`` autobuild session that starts a web server and
a watchdog that automatically rebuilds the documentation when you change an
``*.rst`` file::

    cd docs                          # go to the docs folder
    make livehtml                    # run the autobuild watchdog, ensure that
                                     # there are no syntax errors and warnings
    open http://127.0.0.1:8000       # open the autobuild preview

Here are some useful links summarizing the Sphinx syntax:

- `numpydoc docstring guide <https://numpydoc.readthedocs.io/en/latest/format.html>`_
- `rst cheat sheet <https://github.com/ralsina/rst-cheatsheet/blob/master/rst-cheatsheet.rst>`_
- `rst basics <http://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`_

.. _NumPy docstring format: https://numpydoc.readthedocs.io/en/latest/format.html
.. |WIP| raw:: html

  <abbr title="work in progress">WIP</abbr>
