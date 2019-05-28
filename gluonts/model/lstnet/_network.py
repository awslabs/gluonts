# Standard library imports
from typing import List

# Third-party imports
from mxnet import gluon
from mxnet.gluon import nn, rnn

# First-party imports
from gluonts.core.component import validated
from gluonts.model.common import Tensor


class LSTNetBase(gluon.HybridBlock):
    @validated()
    def __init__(
            self,
            skip: int,
            ar_window: int,
            data_window: int,
            prediction_length: int,
            num_series: int,
            num_samples: int,
            conv_hid: int = 100,
            gru_hid: int = 100,
            skip_gru_hid: int = 5,
            dropout_rate: float = 0.2,
            kernel_size: int = 6,
            **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.prediction_length = prediction_length

        self.num_samples = num_samples

        self.skip = skip
        self.kernel_size = kernel_size
        self.num_series = num_series
        self.ar_window = ar_window
        # calculate output shape of channel assuming padding=0 and step=1
        self.conv_output_shape = data_window - (self.kernel_size - 1)
        # Slice off multiples of skip from convolution output
        self.channels_rounded_to_skip_count = self.conv_output_shape // self.skip * self.skip
        self.skip_by_c_dims = skip_gru_hid * self.skip

        with self.name_scope():
            self.conv = nn.Conv1D(conv_hid, kernel_size=self.kernel_size, layout='NCW',
                                  activation='relu')
            self.dropout = nn.Dropout(dropout_rate)
            self.gru = rnn.GRU(gru_hid, layout='TNC')
            self.skip_gru = rnn.GRU(skip_gru_hid, layout='TNC')
            self.fc = nn.Dense(num_series)
            self.ar_fc = nn.Dense(1)

    # noinspection PyMethodOverriding,PyPep8Naming
    def hybrid_forward(
            self,
            F,
            past_target: Tensor
    ) -> Tensor:
        # Convolution
        # Transpose NTC to to NCT (a.k.a NCW) before convolution
        c = self.conv(past_target.transpose((0, 2, 1)))
        c = self.dropout(c)

        # GRU
        r = self.gru(c.transpose((2, 0, 1)))  # Transpose NCT to TNC before GRU
        r = r.slice_axis(axis=0, begin=-1, end=None).squeeze(axis=0)  # Only keep the last output
        r = self.dropout(r)  # Now in NC layout

        # Skip GRU
        # take only last `channels_rounded_to_skip_count` channels
        skip_c = c.slice_axis(axis=2, begin=-self.channels_rounded_to_skip_count, end=None)

        # Reshape to NCT x skip
        skip_c = skip_c.reshape((0, 0, -4, -1, self.skip))
        # Transpose to T x N x skip x C
        skip_c = skip_c.transpose((2, 0, 3, 1))
        # Reshape to Tx (Nxskip) x C
        skip_c = skip_c.reshape((0, -3, -1))

        s = self.skip_gru(skip_c)

        # Only keep the last output (now in (Nxskip) x C layout)
        s = s.slice_axis(axis=0, begin=-1, end=None).squeeze(axis=0)
        # Now in N x (skipxC) layout
        s = s.reshape((-1, self.skip_by_c_dims))

        # FC layer
        fc = self.fc(F.concat(r, s))  # NC layout

        # Autoregressive highway
        ar_x = past_target.slice_axis(axis=1, begin=-self.ar_window,
                                      end=None)  # NTC layout
        ar_x = ar_x.transpose((0, 2, 1))  # NCT layout
        ar_x = ar_x.reshape((-3, -1))  # (NC) x T layout
        ar = self.ar_fc(ar_x)
        ar = ar.reshape((-1, self.num_series))  # NC layout

        # Add autoregressive and fc outputs
        res = fc + ar
        return res


class LSTNetTrainingNetwork(LSTNetBase):
    @validated()
    def __init__(self, skip: int, ar_window: int, data_window: int, prediction_length: int,
                 num_series: int, num_samples: int, conv_hid: int = 100, gru_hid: int = 100,
                 skip_gru_hid: int = 5, dropout_rate: float = 0.2, kernel_size: int = 6,
                 **kwargs) -> None:
        super().__init__(skip, ar_window, data_window, prediction_length, num_series, num_samples,
                         conv_hid, gru_hid, skip_gru_hid, dropout_rate, kernel_size, **kwargs)

        self.l1 = gluon.loss.L1Loss()


    # noinspection PyMethodOverriding,PyPep8Naming
    def hybrid_forward(
            self,
            F,
            past_target: Tensor,
            future_target: Tensor
    ) -> List[Tensor]:
        res = super().hybrid_forward(F, past_target)
        loss = self.l1(res, future_target)
        return [loss, res]


class LSTNetPredictionNetwork(LSTNetBase):
    # noinspection PyMethodOverriding,PyPep8Naming
    def hybrid_forward(
            self,
            F,
            past_target: Tensor,
    ) -> Tensor:
        res = super().hybrid_forward(F, past_target)

        # Expected output is (batch_size, num_samples, prediction_length)
        # Adding num_samples axis of 1
        return res.expand_dims(axis=1)

