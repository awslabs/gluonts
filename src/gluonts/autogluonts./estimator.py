from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.trainer import Trainer
from gluonts.dataset.loader import TrainDataLoader
import numpy as np
from gluonts.support.util import get_hybrid_forward_input_names
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.model.seq2seq._mq_dnn_estimator import MQCNNEstimator
from gluonts.evaluation import Evaluator
from dataset import dataset
estimator = MQCNNEstimator(
  #  num_hidden_dimensions=[10],
    prediction_length=dataset.metadata.prediction_length,
    context_length=100,
    freq=dataset.metadata.freq,
    trainer=Trainer(ctx="cpu",
                    epochs=5,
                    learning_rate=1e-3,
                    num_batches_per_epoch=100
                   )
)

net = estimator.create_training_network()