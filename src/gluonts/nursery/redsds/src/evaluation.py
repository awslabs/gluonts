import torch
import pandas as pd
import numpy as np
from gluonts.evaluation import Evaluator, MultivariateEvaluator
from gluonts.model.forecast import SampleForecast
from gluonts.transform import TransformedDataset
from gluonts.transform import AdhocTransform
from gluonts.torch.batchify import batchify
from gluonts.dataset.loader import InferenceDataLoader
from sklearn.metrics import f1_score
from sklearn.metrics.cluster import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    contingency_matrix)
from scipy.optimize import linear_sum_assignment

from .datasets import create_input_transform


def numpy2dataframe_gen(array, index):
    for ts in array:
        df = pd.DataFrame(ts, index=index)
        yield df


def numpy2forecast_gen(array, time_stamp, freq, item_id=None):
    for ts in array:
        sf = SampleForecast(
            samples=ts, start_date=time_stamp, freq=freq, item_id=item_id)
        yield sf


def evaluate_forecast(true_futures, forecasts, freq='H'):
    assert (true_futures.shape == forecasts.shape[1:]), (
            'Last three dims of true_futures and forecasts'
            f' should match got {true_futures.shape} and'
            f' {forecasts.shape}')
    B, prediction_len, obs_dim = true_futures.shape
    num_samples, _, _, _ = forecasts.shape
    forecasts = np.swapaxes(forecasts, 0, 1)
    if obs_dim == 1:
        true_futures = true_futures.squeeze(-1)
        forecasts = forecasts.squeeze(-1)
    dummy_time_stamp = pd.Timestamp(year=2021, month=1, day=1)
    idx = pd.date_range(
        start=dummy_time_stamp, periods=prediction_len, freq=freq)
    if obs_dim == 1:
        evaluator = Evaluator(
            quantiles=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9))
    else:
        evaluator = MultivariateEvaluator(
            quantiles=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9))
    agg_metrics, item_metrics = evaluator(
        iter(numpy2dataframe_gen(true_futures, idx)),
        iter(numpy2forecast_gen(forecasts, dummy_time_stamp, freq)),
        num_series=B)
    return agg_metrics


def evaluate_gts_dataset(
        dataset,
        model,
        device=torch.device('cpu'),
        num_samples=100,
        deterministic_z=False,
        deterministic_x=False,
        deterministic_y=False,
        max_len=1000,
        batch_size=10):
    assert dataset.metadata['context_length'] == model.context_length
    assert dataset.metadata['prediction_length'] == model.prediction_length
    freq = dataset.freq
    context_length = model.context_length
    prediction_length = model.prediction_length
    test_dataset = dataset.gluonts_dataset
    if max_len >= len(test_dataset):
        max_len = len(test_dataset)
    else:
        print(f'WARNING: Testing only on {max_len} time-series but '
              f'the size of test set is {len(test_dataset)}!')

    def add_ts_dataframe(data_iterator):
        for data_entry in data_iterator:
            data = data_entry.copy()
            index = pd.date_range(
                start=data["start"],
                freq=freq,
                periods=data["target"].shape[-1],
            )
            data["ts"] = pd.DataFrame(
                index=index, data=data["target"].transpose()
            )
            yield data

    def test_generator(dataset):
        size = 0
        for data_entry in add_ts_dataframe(iter(dataset)):
            if size < max_len:
                yield data_entry["ts"]
                size += 1
            else:
                break

    def truncate_target(data):
        data = data.copy()
        target = data["target"]
        assert (
            target.shape[-1] >= prediction_length
        )  # handles multivariate case (target_dim, history_length)
        data["target"] = target[..., : -prediction_length]
        return data

    trucated_dataset = TransformedDataset(
        test_dataset, transformation=AdhocTransform(truncate_target))
    input_transform = create_input_transform(
        is_train=False,
        prediction_length=prediction_length,
        past_length=context_length,
        use_feat_static_cat=True,
        use_feat_dynamic_real=False,
        freq=freq,
        time_features=None)
    inference_loader = InferenceDataLoader(
        dataset=trucated_dataset,
        transform=input_transform,
        batch_size=batch_size,
        stack_fn=batchify,
        num_workers=1)

    def prediction_generator(inference_loader):
        size = 0
        for batch in inference_loader:
            ys = batch['past_target']
            forecast_starts = batch['forecast_start']
            item_ids = batch['item_id'] if 'item_id' in batch \
                else [int(x[0]) for x in batch['feat_static_cat']]
            assert ys.shape[-2] == context_length

            rec_n_forecast = model.predict(
                ys.to(device),
                ctrl_inputs=dict(
                    feat_static_cat=batch['feat_static_cat'].long().to(device),
                    past_time_feat=batch['past_time_feat'].to(device),
                    future_time_feat=batch['future_time_feat'].to(device),
                ),
                num_samples=num_samples,
                deterministic_z=deterministic_z,
                deterministic_x=deterministic_x,
                deterministic_y=deterministic_y,
            )['rec_n_forecast'].data.cpu().numpy()
            assert (
                rec_n_forecast.shape[-2] ==
                prediction_length + context_length), (
                    f'Length of rec + forecast {rec_n_forecast.shape[-2]}'
                    f' along time dimension should be '
                    f'context_length {context_length} '
                    f'+ prediction_length {prediction_length}')
            forecasts = rec_n_forecast[
                ..., -prediction_length:, :]
            forecasts = np.swapaxes(forecasts, 0, 1)
            if forecasts.shape[-1] == 1:
                forecasts = forecasts.squeeze(-1)

            for forecast_start, item_id, forecast \
                    in zip(forecast_starts, item_ids, forecasts):
                if size < max_len:
                    sf = SampleForecast(
                        samples=forecast,
                        start_date=forecast_start,
                        freq=freq,
                        item_id=item_id)
                    yield sf
                    size += 1
                else:
                    break
            if size == max_len:
                break
    pred_gen = prediction_generator(inference_loader)
    test_gen = test_generator(test_dataset)
    evaluator = Evaluator(
        quantiles=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9))
    agg_metrics, item_metrics = evaluator(
        iter(test_gen),
        iter(pred_gen),
        num_series=max_len)
    return agg_metrics


def segmentation_accuracy(true_seg, predicted_seg):
    unique_labels = np.unique(true_seg)
    unique_preds = np.unique(predicted_seg)
    cont_mat = contingency_matrix(true_seg, predicted_seg)
    cost_matrix = cont_mat.max() - cont_mat
    row_idx, col_idx = linear_sum_assignment(cost_matrix)
    mapper = {unique_preds[c]: unique_labels[r]
              for (r, c) in zip(row_idx, col_idx)}
    remapped_y_pred = np.array(
        [mapper[i] for i in predicted_seg]
    )
    acc = np.sum(true_seg == remapped_y_pred) / true_seg.size
    f1 = f1_score(true_seg, remapped_y_pred, average='weighted')
    return acc, f1


def evaluate_segmentation(true_seg, predicted_seg, K):
    assert true_seg.shape == predicted_seg.shape, (
           'true_seg and predicted_seg shapes do not match.'
           f' Got {true_seg.shape} and {predicted_seg.shape}')
    true_seg = true_seg.reshape(-1)
    predicted_seg = predicted_seg.reshape(-1)
    nmi_score = normalized_mutual_info_score(true_seg, predicted_seg)
    ari_score = adjusted_rand_score(true_seg, predicted_seg)

    if K > true_seg.max() + 1:
        raise NotImplementedError()
    else:
        acc, f1 = segmentation_accuracy(true_seg, predicted_seg)
    metrics = dict(
        nmi_score=nmi_score,
        ari_score=ari_score,
        accuracy=acc,
        f1_score=f1)
    return metrics


if __name__ == '__main__':
    pass
