# Third-party imports
import mxnet as mx

# First-party imports
from gluonts.core.component import validated
from gluonts.distribution import DistributionOutput
from gluonts.model.common import Tensor


class SimpleLSTMNetworkBase(mx.gluon.HybridBlock):
    @validated()
    def __init__(
        self,
        num_layers: int,
        num_cells: int,
        context_length: int,
        prediction_length: int,
        distr_output: DistributionOutput,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.num_cells = num_cells
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.distr_output = distr_output

        with self.name_scope():
            self.distr_args_proj = self.distr_output.get_args_proj()
            self.lstm = mx.gluon.rnn.HybridSequentialRNNCell()
            for k in range(num_layers):
                self.lstm.add(mx.gluon.rnn.LSTMCell(hidden_size=num_cells))


class SimpleLSTMTrainingNetwork(SimpleLSTMNetworkBase):
    # noinspection PyMethodOverriding,PyPep8Naming
    def hybrid_forward(
        self, F, past_target: Tensor, future_target: Tensor
    ) -> Tensor:
        """

        Parameters
        ----------
        F
        past_target : (batch_size, seq_len, target_dim)
        future_target : (batch_size, seq_len, target_dim)

        Returns
        -------

        """

        # (batch_size, context_length - 1, target_dim)
        context_lags = past_target.slice_axis(
            axis=1, begin=0, end=self.context_length - 1
        )

        # (batch_size, prediction_length, target_dim)
        prediction_lags = F.concat(
            past_target.slice_axis(
                axis=1, begin=self.context_length - 1, end=self.context_length
            ),
            future_target.slice_axis(
                axis=1, begin=0, end=self.prediction_length - 1
            ),
            dim=1,
        )

        # unroll encoder in the context range
        _, state = self.lstm.unroll(
            inputs=context_lags,
            length=self.context_length - 1,
            layout="NTC",
            merge_outputs=True,
        )

        # compute parameters in the prediction range
        outputs, _ = self.lstm.unroll(
            inputs=prediction_lags,
            length=self.prediction_length,
            layout="NTC",
            merge_outputs=True,
            begin_state=state,
        )

        distr_args = self.distr_args_proj(outputs)
        distr = self.distr_output.distribution(distr_args)

        # (batch_size, target_dim)
        return distr.loss(future_target).mean(axis=1)


class SimpleLSTMPredictionNetwork(SimpleLSTMNetworkBase):
    @validated()
    def __init__(self, num_sample_paths: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_sample_paths = num_sample_paths

    def sampling_decoder(self, F, input0: Tensor, begin_state=None) -> Tensor:
        """

        Parameters
        ----------
        F
        input0 : first input to the LSTM, with layout N1C (i.e. NTC with T=1).
        begin_state : initial state of the LSTM.

        Returns tensor containing sample paths, with layout NST (where S stands for "samples").
        -------

        """
        sample = input0.repeat(repeats=self.num_sample_paths, axis=0)
        begin_state = [
            s.repeat(repeats=self.num_sample_paths, axis=0)
            for s in begin_state
        ]
        samples = []

        for k in range(self.prediction_length):
            output, begin_state = self.lstm.unroll(
                inputs=sample,
                length=1,
                begin_state=begin_state,
                layout="NTC",
                merge_outputs=True,
            )

            distr_args = self.distr_args_proj(output)
            distr = self.distr_output.distribution(distr_args)
            samples.append(distr.sample())

        return F.concat(*samples, dim=1).reshape(
            shape=(-1, self.num_sample_paths, self.prediction_length)
        )

    # noinspection PyMethodOverriding,PyPep8Naming
    def hybrid_forward(self, F, past_target: Tensor) -> Tensor:
        """

        Parameters
        ----------
        F
        past_target : (batch_size, target_dim, seq_len)

        Returns samples with shape (batch_size, num_sample, prediction_length)
        -------

        """

        # (batch_size, seq_len, target_dim)
        context_lags = past_target.slice_axis(
            axis=1, begin=0, end=self.context_length - 1
        )

        # unroll encoder in the context range
        _, state = self.lstm.unroll(
            inputs=context_lags,
            length=self.context_length - 1,
            layout="NTC",
            merge_outputs=True,
        )

        # take last time point
        input0 = past_target.slice_axis(
            axis=1, begin=self.context_length - 1, end=self.context_length
        )

        return self.sampling_decoder(F, input0, state)
