import os
import numpy as np
import inspect
import mxnet as mx
from gluonts.mx.trainer import learning_rate_scheduler as lrs
from mxnet import gluon, init

import autogluon as ag
from autogluon.utils.mxutils import get_data_rec



from gluonts.gluonts_tqdm import tqdm
from gluonts.support.util import get_hybrid_forward_input_names
from gluonts.model.simple_feedforward._estimator import  SimpleFeedForwardEstimator
from gluonts.model.seq2seq._mq_dnn_estimator import MQCNNEstimator
from gluonts.mx.trainer import Trainer

from dataset import dataset
from eval import evaluation

from functools import partial
from gluonts.mx.batchify import batchify, as_in_context
from gluonts.dataset.loader import TrainDataLoader
from gluonts.mx.trainer import Trainer
from gluonts.mx.trainer import learning_rate_scheduler as lrs




class AutoEstimator:
    #put the dataset on train
    def __init__(self,dictionary_of_hyperparameters):
        '''

        Args:
            model: gluonts.model.estimator
            dictionary_of_hyperparameters: autogluon search space
            dataset: gluotns dataset
        '''
        #default configuration
        default_config = {}
        default_config['model'] = SimpleFeedForwardEstimator
        default_config['metric'] = 'MAPE'

        #Trainer configuration
        default_config['learning_rate'] = 1e-3
        default_config['epochs'] = 100
        default_config['num_batches_per_epoch'] = 100

        #model configuraiton
        default_config['context_length'] = 100
        default_config['freq'] = None
        default_config['prediction_length'] = None

        #Non-essential Hyperparameter for learning_rate scheduler

        default_config['objective'] = "min"
        default_config['patience'] = 10
        default_config['decay_factor'] = 0.5
        default_config['min_lr'] = 1e-5


        # Non-essential Hyperparameter for optimizer
        default_config['weight_decay']=1e-8
        default_config['clip_gradient']= 10


        # Non-essential Hyperparameter for TrainDataLoader

        default_config['dtype'] = np.float32
        default_config['batch_size'] = 100
        default_config['num_batches_per_epoch'] = 50
        default_config['ctx'] = None
        default_config['num_workers'] = None
        default_config['num_prefetch'] = None
        default_config['shuffle_buffer_length'] = None

        self.default_config = default_config
        self.dictionary_of_hyperparameters = dictionary_of_hyperparameters


        for config in default_config.keys():
            if not config in self.dictionary_of_hyperparameters.keys():
                self.dictionary_of_hyperparameters[config] = default_config[config]

    def train(self,dataset_train,dataset_test):
        def train_ts(args, reporter):
            # get variables from args
            learning_rate = args.learning_rate
            epochs = args.epochs
            estimator_parameter_list = []
            estimator_signiture = inspect.signature(args.model.__init__)
            estimator_parameter_list = [parameter for parameter in estimator_signiture.parameters.keys() if parameter != 'self']
            estimator_parameter_register = {}
            for parameter in estimator_parameter_list:
                if parameter in args.keys():
                    estimator_parameter_register[parameter] = args[parameter]
            trainer = Trainer(learning_rate=learning_rate,
                                                     epochs = epochs,
                                                     num_batches_per_epoch=args.num_batches_per_epoch)
            estimator_parameter_register['trainer'] = trainer
            estimator = args.model(**estimator_parameter_register)

            transformation = estimator.create_transformation()
            lr_scheduler = lrs.MetricAttentiveScheduler(
                objective=args.objective,
                patience=args.patience,
                min_lr=args.min_lr,
                decay_factor=args.decay_factor,
            )

            optimizer = mx.optimizer.Adam(
                learning_rate=learning_rate,
                lr_scheduler=lr_scheduler,
                wd=args.weight_decay,
                clip_gradient=args.clip_gradient,
            )

            # datasets and dataloaders

            batch_iter = TrainDataLoader(dataset=dataset_train,
                                         transform=transformation,
                                         batch_size=args.batch_size,
                                         num_batches_per_epoch=args.num_batches_per_epoch,
                                         stack_fn=partial(
                                             batchify, ctx=args.ctx, dtype=args.dtype,
                                         ),
                                         num_workers=args.num_workers,
                                         num_prefetch=args.num_prefetch,
                                         shuffle_buffer_length=args.shuffle_buffer_length,
                                         decode_fn=partial(as_in_context, ctx=args.ctx),
                                         )

            net = estimator.create_training_network()
            net.initialize(ctx=args.ctx, init="xavier")
            trainer = mx.gluon.Trainer(
                net.collect_params(),
                optimizer=optimizer,
                kvstore="device",  # FIXME: initialize properly
            )
            input_names = get_hybrid_forward_input_names(net)

            # Training
            def train_epoch(epoch):
                epoch_loss = mx.metric.Loss()
                with tqdm(batch_iter) as it:
                    for batch_no, data_entry in enumerate(it, start=1):
                        inputs = [data_entry[k] for k in input_names]
                        with mx.autograd.record():
                            output = net(*inputs)
                            loss = output
                        loss.backward()
                        trainer.step(args.batch_size)
                        epoch_loss.update(None, preds=loss)

            for epoch in tqdm(range(0, epochs)):
                train_epoch(epoch)
                result = evaluation(estimator, transformation, net, dataset_test,args.metric)
                reporter(epoch=epoch + 1, accuracy=result)

        @ag.args()
        def train_finetune(args, reporter):
            train_ts(args, reporter)

        train_finetune.register_args(**self.dictionary_of_hyperparameters)
        self.scheduler = ag.scheduler.FIFOScheduler(train_finetune,
                                                 resource={'num_cpus': 4, 'num_gpus': 0},
                                                 num_trials=3,
                                                 time_attr='epoch',
                                                 reward_attr="accuracy")
        self.scheduler.run()
        self.scheduler.join_jobs()
        #self.scheduler.get_training_curves(plot=True,use_legend=False)


    def get_best_estimator(self):
        num_trails = len(self.scheduler.config_history)
        training_history = self.scheduler.training_history
        config_history = self.scheduler.config_history
        config_history = [config_history[str(i)] for i in range(num_trails)]
        rewards = [[training_history[str(i)][j]['accuracy'] for j in range(len(training_history[str(i)]))] for i in range(num_trails)]
        best_rewards = [np.max(rewards[i]) for i in range(num_trails)]
        best_reward = np.max(best_rewards)
        best_id = np.argmax(best_rewards)
        best_adjust_config = config_history[best_id]
        best_config = {'best_reward':best_reward}
        best_config.update((best_adjust_config))
        for config in self.dictionary_of_hyperparameters.keys():
            if not config in best_config.keys():
                best_config[config] = self.dictionary_of_hyperparameters[config]
        self.best_config = best_config

        self.scheduler_best_config = self.scheduler.get_best_config()
        model_choice = best_config['model▁choice']
        model = best_config['model'][model_choice]
        estimator_parameter_list = []
        estimator_signiture = inspect.signature(model.__init__)
        estimator_parameter_list = [parameter for parameter in estimator_signiture.parameters.keys() if
                                    parameter != 'self']
        estimator_parameter_register = {}
        for parameter in estimator_parameter_list:
            if parameter in best_config.keys():
                estimator_parameter_register[parameter] = best_config[parameter]
        trainer = Trainer(learning_rate=best_config['learning_rate'],
                          epochs=best_config['epochs'],
                          num_batches_per_epoch=best_config['num_batches_per_epoch'])
        estimator_parameter_register['trainer'] = trainer
        estimator = model(**estimator_parameter_register)
        return estimator


        #create best estimator through best_config from searcher
        #best_config =  self.scheduler.get_best_config()
        #print(best_config)

        '''
        self.final_estimator = self.dictionary_of_hyperparameters['model'](
            prediction_length=self.dictionary_of_hyperparameters['prediction_length'],
            context_length=best_config['context_length'],
            freq=self.dictionary_of_hyperparameters['freq'],
            trainer=Trainer(ctx=self.dictionary_of_hyperparameters['ctx'],
                            epochs=best_epoch,
                            learning_rate=best_config['learning_rate'],

                            )
            )
        return self.final_estimator
        '''

    def get_best_predictor(self):
        from gluonts.evaluation.backtest import make_evaluation_predictions
        from gluonts.evaluation import Evaluator
        #create best predictor through re-training
        self.predictor = self.final_estimator.train(dataset.train)
        forecast_it, ts_it = make_evaluation_predictions(
            dataset=dataset.test,  # test dataset
            predictor=self.predictor,  # predictor
            num_samples=100,  # number of sample paths we want for evaluation
        )

        forecasts = list(forecast_it)
        tss = list(ts_it)
        evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
        agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts), num_series=len(dataset.test))
        print(agg_metrics[self.metric])

        return  self.predictor

if __name__ == '__main__':
    from dataset import dataset
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
