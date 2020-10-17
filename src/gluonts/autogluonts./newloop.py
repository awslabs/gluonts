# Standard library imports

import time
import gluonts
import mxnet as mx
# Third-party imports
import mxnet.autograd as autograd
import numpy as np
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.mx.trainer.model_averaging import AveragingStrategy
from gluonts.mx.trainer.model_iteration_averaging import IterationAveragingStrategy
from gluonts.mx.trainer._base import loss_value
# Relative imports

from metric_result import *

epochs = 10
#net.initialize(ctx=None, init='xavier')

avg_strategy = AveragingStrategy()
def newloop(
        epoch_no, estimator, net,trainer,inputs,metric,dataset_test,is_training: bool = True,
batch_size = 32):

    tic = time.time()

    epoch_loss = mx.metric.Loss()

    # use averaged model for validation
    if not is_training and isinstance(
            avg_strategy, IterationAveragingStrategy
    ):
        avg_strategy.load_averaged_model(net)



    with mx.autograd.record():
        output = net(*inputs)
        # network can returns several outputs, the first being always the loss
        # when having multiple outputs, the forward returns a list in the case of hybrid and a
        # tuple otherwise
        # we may wrap network outputs in the future to avoid this type check
        if isinstance(output, (list, tuple)):
            loss = output[0]
        else:
            loss = output

    if is_training:
        loss.backward()
        trainer.step(batch_size)


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
    dataset=dataset_test,  # test dataset
    predictor=pre,  # predictor
   num_samples=100,)  # number of sample paths we want for evaluation



    forecasts = list(forecast_it)
    tss = list(ts_it)



    result = get_result(iter(tss), iter(forecasts), metric,len(dataset_test.test))
    return result






