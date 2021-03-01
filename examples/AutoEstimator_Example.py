from gluonts.model.simple_feedforward._estimator import  SimpleFeedForwardEstimator
from gluonts.model.seq2seq._mq_dnn_estimator import MQCNNEstimator
import autogluon as ag
from dataset import dataset
from automodel import AutoEstimator

from gluonts.dataset.artificial import ComplexSeasonalTimeSeries

dataset = ComplexSeasonalTimeSeries(
    num_series=10,
    prediction_length=21,
    freq_str="H",
    length_low=30,
    length_high=200,
    min_val=-10000,
    max_val=10000,
    is_integer=False,
    proportion_missing_values=0,
    is_noise=True,
    is_scale=True,
    percentage_unique_timestamps=1,
    is_out_of_bounds_date=True,
)
from gluonts.dataset.repository.datasets import get_dataset, dataset_recipes
dataset_m4 = get_dataset("m4_hourly", regenerate=False)

prediction_length = dataset.metadata.prediction_length
dictionary_of_hyperparameters = {}

dictionary_of_hyperparameters['model'] = ag.Categorical(SimpleFeedForwardEstimator,MQCNNEstimator)
dictionary_of_hyperparameters['metric'] = 'MAPE'

#Trainer Hyperparameters
dictionary_of_hyperparameters['epochs']  = 5
dictionary_of_hyperparameters['learning_rate'] = ag.Real(1e-3, 1e-2, log=True)

dictionary_of_hyperparameters['freq'] = dataset.metadata.freq
dictionary_of_hyperparameters['prediction_length'] = dataset.metadata.prediction_length
dictionary_of_hyperparameters['context_length'] = ag.Int(2 * prediction_length, 10 * prediction_length)
#deepar, ssf, mqcnn in terms of wQuantileLoss
estimator = AutoEstimator(dictionary_of_hyperparameters)

estimator.train(dataset_train=dataset.train,dataset_test=dataset.test)
# best_estimator = estimator.get_best_estimator()