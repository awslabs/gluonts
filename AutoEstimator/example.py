from gluonts.model.simple_feedforward._estimator import  SimpleFeedForwardEstimator
from gluonts.model.seq2seq._mq_dnn_estimator import MQCNNEstimator
import autogluon as ag
from dataset import dataset
from automodel import AutoEstimator

from dataset import dataset
prediction_length = dataset.metadata.prediction_length

#create dictionary of hyperparameter 
dictionary_of_hyperparameters = {}

#Model and Metric are normally neccesary: default model:SimpleFeedForwardEstimator, default metric: MAPE
dictionary_of_hyperparameters['model'] = ag.Categorical(SimpleFeedForwardEstimator,MQCNNEstimator)
dictionary_of_hyperparameters['metric'] = 'MAPE'

#Trainer Hyperparameters
dictionary_of_hyperparameters['epochs']  = 5
dictionary_of_hyperparameters['learning_rate'] = ag.Real(1e-3, 1e-2, log=True)

#Dataset Hyperparameters: usually unmutable, except for context length in this case
dictionary_of_hyperparameters['freq'] = dataset.metadata.freq
dictionary_of_hyperparameters['prediction_length'] = dataset.metadata.prediction_length
dictionary_of_hyperparameters['context_length'] = ag.Int(2 * prediction_length, 10 * prediction_length)

#create a new estimator by register AutoEstimator with a dictionary of hyperparameter 
estimator = AutoEstimator(dictionary_of_hyperparameters)

#train with train data and test data
estimator.train(dataset_train=dataset.train,dataset_test=dataset.test)
# best_estimator = estimator.get_best_estimator()
