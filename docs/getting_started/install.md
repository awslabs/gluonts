
# Installation

GluonTS is available from PyPi via:

```sh
pip install gluonts
````

```{attention}
**GluonTS uses a minimal dependency model.**

This means that to use most models and features additional dependencies need to
be installed. See the next section for more information.

```

## Optional and Extra Dependencies

Python has the notion of [extras](https://peps.python.org/pep-0508/#extras)
-- dependencies that can be optionally installed to unlock certain features of
a pacakge.

When installing a package, they are passed via ``[...]`` after the package
name:

```sh
pip install some-package[extra-1,extra-2]
````

We make extensive use of optional dependencies in GluonTS to keep the amount of
required dependencies minimal. To still allow users to opt-in to certain
features, we expose many extra dependencies.

For example, we offer support for reading and writing Arrow and Parquet based
datasets using [Apache Arrow](https://arrow.apache.org/). However, it is a
hefty dependency to require, especially if one has no need for it. Thus, we
offer the ``arrow``-extra, which installs the required packages and can be
simply enabled using:

```sh
pip install gluonts[arrow]
````

### Models

You can enable or disable extra dependencies as you prefer, depending on what GluonTS features you are interested in enabling.

* `mxnet` - MXNet-based models
* `torch` - PyTorch-based models
* `R` - R-based models
* `Prophet` - Prophet-based models


### Datasets

* `arrow` - Arrow and Parquet dataset support
* `pro` - bundles `arrow` plus `orjson` for faster datasets


### Other

* `shell` for integration with SageMaker


## Install from Dev Branch


If you are interested in trying out features on dev branch that hasn't been released yet, you have
the option of installing from dev branch directly.


## Install from GitHub


Use the following command to automatically download and install the current code on dev branch:

```sh
pip install git+https://github.com/awslabs/gluon-ts.git
````

## Install from Source Code

You can also first check out the code locally using Git:

.. code-block:: console

   git clone https://github.com/awslabs/gluon-ts
   cd gluon-ts

then use the provided `setup.py` to install into site-packages:

.. code-block:: console

   python setup.py install


.. note::

   You may need to use `sudo` in case you run into permission denied error.


Alternatively, you can set up the package with development mode, so that local changes are
immediately reflected in the installed python package

.. code-block:: console

   python setup.py develop

.. note::

   The dev branch may rely on MXNet nightly builds which are available on PyPI,
   please refer to `this page <http://beta.mxnet.io/install.html>`_ for installation guide.
