import cProfile
import pstats
from pstats import SortKey
import io
from data_store import *
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.model.deepar import DeepAREstimator
from gluonts.model.seq2seq import MQCNNEstimator
from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.model.wavenet import WaveNetEstimator
from run_data_loader import data_loader
import numpy as np

datasets = ['m4_daily', 'm4_hourly', 'm4_weekly', 'solar-energy', 'electricity']
for ds in datasets:
    dataset = get_dataset(ds)
    freq = dataset.metadata.freq
    prediction_length = dataset.metadata.prediction_length
    deepar = DeepAREstimator(
            freq=freq,
            prediction_length=prediction_length,
        )

    mqcnn = MQCNNEstimator(
            freq=freq,
            prediction_length=prediction_length,
        )

    sff = SimpleFeedForwardEstimator(
            num_hidden_dimensions=[10],
            context_length=100,
            freq=freq,
            prediction_length=prediction_length,
        )

    wn = WaveNetEstimator(
            freq=freq,
            prediction_length=prediction_length,
        )

    num_batches_per_epoch = 10
    bin_edges = np.array([-1e20, -1e10, 1, 1e20])
    estimators = [deepar, mqcnn, sff, wn]
    batch = [32]
    for est in estimators:
        for b in batch:
            cProfile.run('data_loader(est, ds, b)', 'restats')
            p = pstats.Stats('restats')
            p.strip_dirs().sort_stats(SortKey.FILENAME)
            result = io.StringIO()
            pstats.Stats('restats', stream=result).print_stats()
            result = result.getvalue()

            # chop the string into a csv-like buffer
            result = 'ncalls' + result.split('ncalls')[-1]
            result = '\n'.join([','.join(line.rstrip().split(None, 5) + [ds, est.__class__.__name__]) for line in result.split('\n') if 'transform' in line[-20:]])

            ###################################
            # title will repeat in each iteration of the for loop
            # result_dup = result[:]
            # title = result_dup.split('\n')[6]
            # title = title.rstrip().split(None, 5)
            # title[-1] = 'filename'
            # title.extend(['dataset', 'estimator'])
            # title = ','.join(title)
            # result = title + '\n' + result
            ###################################

            # save it to disk
            # store_pd('hi.pkl', result)
            store_csv('helloworld.csv', result)




