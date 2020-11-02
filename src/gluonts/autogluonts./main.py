from dataset import dataset,dataset_m4
from gluonts.model.simple_feedforward._estimator import SimpleFeedForwardEstimator
from gluonts.model.seq2seq._mq_dnn_estimator import MQCNNEstimator
from automodel import AutoEstimator
import autogluon as ag
dictionary_of_hyperparameters = {}
dictionary_of_hyperparameters ['learning_rate'] = ag.Real(1e-3, 1e-2, log=True)
dictionary_of_hyperparameters['epochs']=ag.Choice(20,30)

a = AutoEstimator(SimpleFeedForwardEstimator,dictionary_of_hyperparameters,'MSE',dataset_m4)
a.train()
a.scheduler.get_training_curves(plot=True,use_legend = True)
a.get_best_estimator()
print('the best config and reward:', a.best_config)
print('scheduler best config and reward', a.scheduler_best_config, a.scheduler.get_best_reward())