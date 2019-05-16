Installation
============
GluonTS requires Python 3.6 or higher to run. This installation guide assumes that the ``pip`` command
is linked to a Python 3.6 version or higher. We recommend using pyenv_ or virtualenv_ for managing
Python versions.

To install GluonTS, ``git clone`` the repository on your machine, then navigate
to its directory and::

    pip install -e .

This will automatically install all the required dependencies.
Note that the ``-e`` option allows you to update GluonTS by simply doing
``git pull`` in the repository.

.. _pyenv: https://github.com/pyenv/pyenv
.. _virtualenv: https://virtualenv.pypa.io/en/latest/
