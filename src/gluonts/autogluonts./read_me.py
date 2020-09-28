from automodel import AutoEstimator
import autogluon as ag
from dataset import dataset
from gluonts.model.seq2seq._mq_dnn_estimator import MQCNNEstimator

dictionary_of_hyperparameters['epochs']=ag.Choice(1,4)
import numpy as np
#create a auto estimator based on mqcnn
autoMQCNN = AutoEstimator(MQCNNEstimator,dictionary_of_hyperparameters,dataset)
autoMQCNN .train()
