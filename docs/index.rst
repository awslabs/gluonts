Welcome to GluonTS!
===================

GluonTS is a Python package for probabilistic time series modeling, focusing on deep learning models.

Features
--------

* State-of-the-art models implemented with `MXNet <https://mxnet.incubator.apache.org/>`_ and `PyTorch <https://pytorch.org/>`_
* Easy AWS integration via `Amazon SageMaker <https://aws.amazon.com/de/sagemaker/>`_
* Utilities for loading and iterating over time series datasets
* Utilities to evaluate models performance and compare their accuracy
* Building blocks to define custom models and quickly experiment

Installation
------------

GluonTS requires Python 3.7+, and the easiest way to install it is via ``pip``:

.. code-block:: bash
   
   pip install --upgrade gluonts mxnet~=1.8   # to be able to use MXNet-based models
   pip install --upgrade gluonts torch~=1.10  # to be able to use PyTorch-based models

See :doc:`install` for more detailed installation instructions, including optional dependencies.
See :doc:`community/devsetup` for setup instructions in case you want to develop GluonTS.

.. toctree::
   :name: Getting started
   :caption: Getting started
   :maxdepth: 1
   :hidden:

   install

.. toctree::
   :name: Tutorials
   :caption: Tutorials
   :maxdepth: 1
   :hidden:

   tutorials/forecasting/quick_start_tutorial
   tutorials/forecasting/extended_tutorial
   tutorials/data_manipulation/dataframesdataset
   tutorials/data_manipulation/synthetic_data_generation
   tutorials/advanced_topics/trainer_callbacks
   tutorials/advanced_topics/hp_tuning_with_optuna
   tutorials/advanced_topics/howto_pytorch_lightning

.. toctree::
   :name: API docs
   :caption: API docs
   :maxdepth: 1
   :hidden:

   api/gluonts/gluonts

.. toctree::
   :name: Developing
   :caption: Developing
   :maxdepth: 1
   :hidden:

   community/contribute
   community/devsetup
