# Third-party imports
import numpy as np

# First-party imports
from gluonts import transform
from gluonts.dataset.common import ListDataset
from gluonts.model.seq2seq._transform import ForkingSequenceSplitter
from gluonts.transform import TestSplitSampler


def test_forking_sequence_splitter() -> None:
    def make_dataset(N, train_length):
        # generates 2 ** N - 1 timeseries with constant increasing values
        n = 2 ** N - 1

        targets = np.arange(n * train_length).reshape((n, train_length))

        return ListDataset(
            [
                {'start': '2012-01-01', 'target': targets[i, :]}
                for i in range(n)
            ],
            freq='D',
        )

    ds = make_dataset(1, 20)

    trans = transform.Chain(
        trans=[
            transform.AddAgeFeature(
                target_field=transform.FieldName.TARGET,
                output_field='age',
                pred_length=10,
            ),
            ForkingSequenceSplitter(
                train_sampler=TestSplitSampler(),
                time_series_fields=['age'],
                enc_len=5,
                dec_len=3,
            ),
        ]
    )

    out = trans(iter(ds), is_train=True)
    transformed_data = next(iter(out))

    future_target = np.array(
        [
            [13.0, 14.0, 15.0],
            [14.0, 15.0, 16.0],
            [15.0, 16.0, 17.0],
            [16.0, 17.0, 18.0],
            [17.0, 18.0, 19.0],
        ]
    )

    assert (
        np.linalg.norm(future_target - transformed_data['future_target'])
        < 1e-5
    ), "the forking sequence target should be computed correctly."

    trans_oob = transform.Chain(
        trans=[
            transform.AddAgeFeature(
                target_field=transform.FieldName.TARGET,
                output_field='age',
                pred_length=10,
            ),
            ForkingSequenceSplitter(
                train_sampler=TestSplitSampler(),
                time_series_fields=['age'],
                enc_len=20,
                dec_len=20,
            ),
        ]
    )

    transformed_data_oob = next(iter(trans_oob(iter(ds), is_train=True)))

    assert (
        np.sum(transformed_data_oob['future_target']) - np.sum(np.arange(20))
        < 1e-5
    ), "the forking sequence target should be computed correctly."
