from automodel import AutoEstimator
import autogluon as ag
from dataset import dataset
from gluonts.model.simple_feedforward._estimator import  SimpleFeedForwardEstimator

dictionary_of_hyperparameters['epochs']=ag.Choice(1,4)
import numpy as np
#create a auto estimator based on mqcnn
autoSSF = AutoEstimator(SimpleFeedForwardEstimator,dictionary_of_hyperparameters,'MSE',dataset)
autoSSF.train()

