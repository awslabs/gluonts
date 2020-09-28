# Standard library imports
import logging
import os
import tempfile
import time
import uuid
from typing import Any, List, Optional, Union
import gluonts
# Third-party imports
import mxnet as mx
import mxnet.autograd as autograd
import mxnet.gluon.nn as nn
import numpy as np

# First-party imports
from gluonts.core.component import validated
from gluonts.core.exception import GluonTSDataError, GluonTSUserError
from gluonts.dataset.loader import TrainDataLoader, ValidationDataLoader
from gluonts.gluonts_tqdm import tqdm

from gluonts.support.util import HybridContext

# Relative imports
from gluonts.trainer import learning_rate_scheduler as lrs
from asset import *
from getMSE import *
from dataset import dataset
epochs = 10
#net.initialize(ctx=None, init='xavier')
#trainer = mx.gluon.Trainer(
#                    net.collect_params(),
#                    optimizer=optimizer,
#                    kvstore="device",  # FIXME: initialize properly
#                )
avg_strategy = AveragingStrategy()
def newloop(
        epoch_no, estimator, net,trainer,inputs,is_training: bool = True,
):
 #   print('check 0')
    tic = time.time()

    epoch_loss = mx.metric.Loss()

    # use averaged model for validation
    if not is_training and isinstance(
            avg_strategy, IterationAveragingStrategy
    ):
        avg_strategy.load_averaged_model(net)

 #   print('check 1 test')

    with mx.autograd.record():
        output = net(*inputs)
      #  print('check 2.2')
        # network can returns several outputs, the first being always the loss
        # when having multiple outputs, the forward returns a list in the case of hybrid and a
        # tuple otherwise
        # we may wrap network outputs in the future to avoid this type check
        if isinstance(output, (list, tuple)):
            loss = output[0]
       #     print('check 2.3')
        else:
            loss = output
      #  print('check 2')
    if is_training:
       # print(3.1)
        loss.backward()
      #  print(3.1)
        trainer.step(batch_size)
      #  print(3.2)

        # iteration averaging in training
        if isinstance(
                avg_strategy,
                IterationAveragingStrategy,
        ):
            avg_strategy.apply(net)

    epoch_loss.update(None, preds=loss)



    transformation = estimator.create_transformation()
    lv = loss_value(epoch_loss)
    pre = estimator.create_predictor(transformation,net)



    forecast_it, ts_it = make_evaluation_predictions(
    dataset=dataset.test,  # test dataset
    predictor=pre,  # predictor
   num_samples=100,)  # number of sample paths we want for evaluation



    forecasts = list(forecast_it)
    tss = list(ts_it)

    if type(forecasts[0]) == gluonts.model.forecast.QuantileForecast:
        quantile = True
    else:
        quantile = False


    m = call(iter(tss), iter(forecasts), quantile,len(dataset.test))
    return m





    # print out parameters of the network at the first pass

