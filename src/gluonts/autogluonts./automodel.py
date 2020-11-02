import os
import numpy as np

import mxnet as mx
from gluonts.trainer import learning_rate_scheduler as lrs
from mxnet import gluon, init
from newloop import newloop
import autogluon as ag
from autogluon.utils.mxutils import get_data_rec



from gluonts.gluonts_tqdm import tqdm
input_names = ['past_target', 'future_target']
from gluonts.model.simple_feedforward._estimator import  SimpleFeedForwardEstimator
from gluonts.model.seq2seq._mq_dnn_estimator import MQCNNEstimator
from dataset import dataset

from asset import optimizer
from gluonts.dataset.loader import TrainDataLoader
from gluonts.trainer import Trainer



dictionary_of_hyperparameters = {}
dictionary_of_hyperparameters ['learning_rate'] = ag.Real(1e-3, 1e-2, log=True)
dictionary_of_hyperparameters['epochs']=ag.Choice(50,100)

class AutoEstimator:
    def __init__(self,model,dictionary_of_hyperparameters,metric,dataset):
        '''

        Args:
            model: gluonts.model.estimator
            dictionary_of_hyperparameters: autogluon search space
            dataset: gluotns dataset
        '''
        self.model = model
        self.metric = metric
        search_config = {}
        search_config['learning_rate'] = ag.Real(1e-3, 1e-2, log=True)
        search_config['epochs'] = ag.Choice(40, 80)
        self.dictionary_of_hyperparameters = dictionary_of_hyperparameters
        self.record = {}
        for config in search_config.keys():
            if not config in self.dictionary_of_hyperparameters .keys():
                self.dictionary_of_hyperparameters [config] = search_config[config]

        self.prediction_length = 21
        self.dataset = dataset
        self.init_estimator  = self.model(
           # num_hidden_dimensions=[10],
            prediction_length=self.prediction_length,
            context_length=100,
            freq=self.dataset.metadata.freq,
            trainer=Trainer(ctx="cpu",
                            epochs=5,
                            learning_rate=1e-3,
                            num_batches_per_epoch=100
                           )
            )
        transformation = self.init_estimator.create_transformation()
        dtype = np.float32
        num_workers = None
        num_prefetch = None
        shuffle_buffer_length = None
        init_trainer = Trainer(ctx="cpu",
                          epochs=5,
                          learning_rate=0.01,
                          num_batches_per_epoch=100
                          )
        self.training_data_loader = TrainDataLoader(
            dataset=self.dataset.train,
            transform=transformation,
            batch_size=init_trainer.batch_size,
            num_batches_per_epoch=init_trainer.num_batches_per_epoch,
            ctx=init_trainer.ctx,
            dtype=dtype,
            num_workers=num_workers,
            num_prefetch=num_prefetch,
        )


    def train(self):
        #let autogluon.searcher search for best config
        model = self.model
        metric = self.metric
        test_data = self.dataset.test
        prediction_length = self.prediction_length
        with tqdm(self.training_data_loader) as it:
            for batch_no, data_entry in enumerate(it, start=1):
                if False:
                    break
            inputs = [data_entry[k] for k in input_names]
        @ag.args()
        def train_finetune(args, reporter):
            new_estimator = model(
            #    num_hidden_dimensions=[10],
                prediction_length=prediction_length,
                context_length=100,
                freq=dataset.metadata.freq,
                trainer=Trainer(ctx="cpu",
                                epochs=args.epochs,
                                learning_rate=args.learning_rate,
                                num_batches_per_epoch=100
                                )
            )
            print(new_estimator)
            new_net = new_estimator.create_training_network()
            new_net.initialize(ctx=None, init='xavier')
            lr_scheduler = lrs.MetricAttentiveScheduler(
                objective="min",
                patience=new_estimator.trainer.patience,
                decay_factor=new_estimator.trainer.learning_rate_decay_factor,
                min_lr=new_estimator.trainer.minimum_learning_rate,
            )
            optimizer = mx.optimizer.Adam(
                learning_rate=new_estimator.trainer.learning_rate,
                lr_scheduler=lr_scheduler,
                wd=new_estimator.trainer.weight_decay,
                clip_gradient=new_estimator.trainer.clip_gradient,
            )
            trainer = mx.gluon.Trainer(
                new_net.collect_params(),
                optimizer=optimizer,
                kvstore="device",  # FIXME: initialize properly
            )

            for epoch in range(args.epochs):

                result = newloop(epoch,new_estimator,new_net,trainer,inputs,test_data,metric)

                #self.record[args] = net.collect_params()._params
                reporter(epoch = epoch+1, accuracy = -result)


        train_finetune.register_args(**self.dictionary_of_hyperparameters)
        self.scheduler = ag.scheduler.FIFOScheduler(train_finetune,
                                                 resource={'num_cpus': 4, 'num_gpus': 0},
                                                 num_trials=5,
                                                 time_attr='epoch',
                                                 reward_attr="accuracy")
        self.scheduler.run()
        self.scheduler.join_jobs()
        #self.scheduler.get_training_curves(plot=True,use_legend=False)


    def get_best_estimator(self):
        num_trails = 5
        training_history = self.scheduler.training_history
        config_history = self.scheduler.config_history
        lr_history = [config_history[str(i)]['learning_rate'] for i in range(num_trails)]
        rewards = [[training_history[str(i)][j]['accuracy'] for j in range(len(training_history[str(i)]))] for i in range(num_trails)]
        best_rewards = [np.max(rewards[i]) for i in range(num_trails)]
        best_epochs = [np.argmax(rewards[i]) for i in range(num_trails)]
        best_reward = np.max(best_rewards)
        best_id = np.argmax(best_rewards)
        best_epoch = best_epochs[best_id]
        best_lr = lr_history[best_id]
        self.best_config = {'best_epoch':best_epoch,'best_lr':best_lr,'best_reward':best_reward}
        self.scheduler_best_config = self.scheduler.get_best_config()


        #create best estimator through best_config from searcher
        best_config =  self.scheduler.get_best_config()
        #print(best_config)
        self.final_estimator = self.model(
          #  num_hidden_dimensions=[10],
            prediction_length=dataset.metadata.prediction_length,
            context_length=100,
            freq=dataset.metadata.freq,
            trainer=Trainer(ctx="cpu",
                            epochs=best_epoch,
                            learning_rate=best_lr,
                            num_batches_per_epoch=100
                            )
            )
        return self.final_estimator

    def get_best_predictor(self):
        #create best predictor through re-training
        return  self.final_estimator.train(dataset.train)

