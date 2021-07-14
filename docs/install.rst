:orphan:

Installation
~~~~~~~~~~~~

GluonTS relies on the recent version of MXNet. The easiest way to install MXNet
is through `pip <https://pip.pypa.io/en/stable/installing/>`_. The following
command installs the latest version of MXNet.

.. code-block:: console

   pip install --upgrade mxnet~=1.7

.. note::

   There are other pre-build MXNet packages that enable GPU supports and
   accelerate CPU performance, please refer to `this page
   <http://beta.mxnet.io/install.html>`_ for details. Some
   training scripts are recommended to run on GPUs, if you don't have a GPU
   machine at hand, you may consider `running on AWS
   <http://d2l.ai/chapter_appendix/aws.html>`_.


After installing MXNet, you can install the GluonTS toolkit by

.. code-block:: console

   pip install gluonts


Install from Master Branch
~~~~~~~~~~~~~~~~~~~~~~~~~~

If you are interested in trying out features on master branch that hasn't been released yet, you have
the option of installing from master branch directly.


Install from GitHub
+++++++++++++++++++

Use the following command to automatically download and install the current code on master branch:

.. code-block:: console

   pip install git+https://github.com/awslabs/gluon-ts.git


Install from Source Code
++++++++++++++++++++++++

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

   The master branch may rely on MXNet nightly builds which are available on PyPI,
   please refer to `this page <http://beta.mxnet.io/install.html>`_ for installation guide.
