
gluonts.model
~~~~~~~~~~~~~

.. autoclass:: gluonts.model.Estimator
    :members:


.. autoclass:: gluonts.model.Predictor
    :members:


PyTorch-Based Models
====================

To run these models, ensure that you install the required dependencies via:

    pip install gluonts[torch]


DeepAR
------

.. autoclass:: gluonts.torch.model.deepar.DeepAREstimator
    :show-inheritance:


MQF2MultiHorizon
----------------

.. autoclass:: gluonts.torch.model.mqf2.MQF2MultiHorizonEstimator
    :show-inheritance:


SimpleFeedForward
-----------------

.. autoclass:: gluonts.torch.model.simple_feedforward.SimpleFeedForwardEstimator
    :show-inheritance:


MXNet-Based Models
==================

To run these models, ensure that you install the required dependencies via:

    pip install gluonts[mxnet]


DeepAR
------

.. autoclass:: gluonts.model.deepar.DeepAREstimator
    :show-inheritance:


DeepState
---------

.. autoclass:: gluonts.model.deepstate.DeepStateEstimator
    :show-inheritance:


DeepVAR
-------

.. autoclass:: gluonts.model.deepvar.DeepVAREstimator
    :show-inheritance:


DeepVARHierarchical
-------------------

.. autoclass:: gluonts.model.deepvar_hierarchical.DeepVARHierarchicalEstimator
    :show-inheritance:


GaussianProcess
---------------

.. autoclass:: gluonts.model.gp_forecaster.GaussianProcessEstimator
    :show-inheritance:


GPVAR
-----

.. autoclass:: gluonts.model.gpvar.GPVAREstimator
    :show-inheritance:


LSTNet
------

.. autoclass:: gluonts.model.lstnet.LSTNetEstimator
    :show-inheritance:


NBEATS
------

.. autoclass:: gluonts.model.n_beats.NBEATSEstimator
    :show-inheritance:


SelfAttention
-------------

.. autoclass:: gluonts.model.san.SelfAttentionEstimator
    :show-inheritance:


Sequence to Sequence
--------------------

.. autoclass:: gluonts.model.seq2seq.MQCNNEstimator
    :show-inheritance:

.. autoclass:: gluonts.model.seq2seq.MQRNNEstimator
    :show-inheritance:

.. autoclass:: gluonts.model.seq2seq.Seq2SeqEstimator
    :show-inheritance:


SimpleFeedForward
-----------------

.. autoclass:: gluonts.model.simple_feedforward.SimpleFeedForwardEstimator
    :show-inheritance:


TemporalFusionTransformer
-------------------------


.. autoclass:: gluonts.model.tft.TemporalFusionTransformerEstimator
    :show-inheritance:


Temporal Point Process
----------------------

.. autoclass:: gluonts.model.tpp.DeepTPPEstimator
    :show-inheritance:


Transformer
-----------

.. autoclass:: gluonts.model.transformer.TransformerEstimator
    :show-inheritance:


WaveNet
-------

.. autoclass:: gluonts.model.wavenet.WaveNetEstimator
    :show-inheritance:


DeepRenewalProcess
------------------

.. autoclass:: gluonts.model.renewal.DeepRenewalProcessEstimator
    :show-inheritance:


External Models
===============

Prophet
-------

.. autoclass:: gluonts.model.prophet.ProphetPredictor
    :show-inheritance:


R-Forecast
----------

.. autoclass:: gluonts.model.r_forecast.RForecastPredictor
    :show-inheritance:


Other
=====

Naive2
------

.. autoclass:: gluonts.model.naive_2.Naive2Predictor


NPTS
----

.. autoclass:: gluonts.model.npts.NPTSPredictor


Rotbaum
-------

.. autoclass:: gluonts.model.rotbaum.TreeEstimator



SeasonalNaive
-------------

.. autoclass:: gluonts.model.seasonal_naive.SeasonalNaivePredictor



Trivial
-------

.. autoclass:: gluonts.model.trivial.constant.ConstantPredictor

.. autoclass:: gluonts.model.trivial.constant.ConstantValuePredictor

.. autoclass:: gluonts.model.trivial.identity.IdentityPredictor

.. autoclass:: gluonts.model.trivial.mean.MeanEstimator

.. autoclass:: gluonts.model.trivial.mean.MeanPredictor

.. autoclass:: gluonts.model.trivial.mean.MovingAveragePredictor
