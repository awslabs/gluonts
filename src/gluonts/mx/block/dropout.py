# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

from typing import Tuple

from mxnet.gluon.rnn import (
    BidirectionalCell,
    ModifierCell,
    RecurrentCell,
    SequentialRNNCell,
)

from gluonts.core.component import validated
from gluonts.mx import Tensor


class VariationalZoneoutCell(ModifierCell):
    """
    Applies Variational Zoneout on base cell.
    The implementation follows [GG16]_.
    Variational zoneout uses the same mask across time-steps. It can be applied to RNN
    outputs, and states. The masks for them are not shared.

    The mask is initialized when stepping forward for the first time and will remain
    the same until .reset() is called. Thus, if using the cell and stepping manually without calling
    .unroll(), the .reset() should be called after each sequence.

    Parameters
    ----------
    base_cell
        The cell on which to perform variational dropout.
    zoneout_outputs
        The dropout rate for outputs. Won't apply dropout if it equals 0.
    zoneout_states
        The dropout rate for state inputs on the first state channel.
        Won't apply dropout if it equals 0.

    """

    @validated()
    def __init__(
        self,
        base_cell: RecurrentCell,
        zoneout_outputs: float = 0.0,
        zoneout_states: float = 0.0,
    ):
        assert not isinstance(base_cell, BidirectionalCell), (
            "BidirectionalCell doesn't support zoneout since it doesn't support step. "
            "Please add VariationalZoneoutCell to the cells underneath instead."
        )
        assert (
            not isinstance(base_cell, SequentialRNNCell)
            or not base_cell._bidirectional
        ), (
            "Bidirectional SequentialRNNCell doesn't support zoneout. "
            "Please add VariationalZoneoutCell to the cells underneath instead."
        )
        super(VariationalZoneoutCell, self).__init__(base_cell)
        self.zoneout_outputs = zoneout_outputs
        self.zoneout_states = zoneout_states
        self._prev_output = None

        # shared masks across time-steps
        self.zoneout_states_mask = None
        self.zoneout_outputs_mask = None

    def __repr__(self):
        s = "{name}(p_out={zoneout_outputs}, p_state={zoneout_states}, {base_cell})"
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def _alias(self):
        return "variationalzoneout"

    def reset(self):
        super(VariationalZoneoutCell, self).reset()
        self._prev_output = None

        self.zoneout_states_mask = None
        self.zoneout_outputs_mask = None

    def _initialize_states_masks(self, F, states):
        if self.zoneout_states and self.zoneout_states_mask is None:
            self.zoneout_states_mask = [
                F.Dropout(F.ones_like(state), p=self.zoneout_states)
                for state in states
            ]

    def _initialize_outputs_mask(self, F, output):
        if self.zoneout_outputs and self.zoneout_outputs_mask is None:
            self.zoneout_outputs_mask = F.Dropout(
                F.ones_like(output), p=self.zoneout_outputs
            )

    def hybrid_forward(
        self, F, inputs: Tensor, states: Tensor
    ) -> Tuple[Tensor, Tensor]:
        cell, p_outputs, p_states = (
            self.base_cell,
            self.zoneout_outputs,
            self.zoneout_states,
        )
        next_output, next_states = cell(inputs, states)

        prev_output = self._prev_output
        if prev_output is None:
            prev_output = F.zeros_like(next_output)

        self._initialize_outputs_mask(F, next_output)

        output = (
            F.where(self.zoneout_outputs_mask, next_output, prev_output)
            if p_outputs != 0.0
            else next_output
        )

        self._initialize_states_masks(F, next_states)
        assert self.zoneout_states_mask is not None

        new_states = (
            [
                F.where(state_mask, new_s, old_s)
                for state_mask, new_s, old_s in zip(
                    self.zoneout_states_mask, next_states, states
                )
            ]
            if p_states != 0.0
            else next_states
        )

        self._prev_output = output

        return output, new_states


class RNNZoneoutCell(ModifierCell):
    """
    Applies Zoneout on base cell.
    The implementation follows [KMK16]_.
    Compared to mx.gluon.rnn.ZoneoutCell, this implementation uses the same mask for output and states[0],
    since for RNN cells, states[0] is the same as output, except for ResidualCell, where states[0] = input + ouptut

    Parameters
    ----------
    base_cell
        The cell on which to perform variational dropout.
    zoneout_outputs
        The dropout rate for outputs. Won't apply dropout if it equals 0.
    zoneout_states
        The dropout rate for state inputs on the first state channel.
        Won't apply dropout if it equals 0.

    """

    @validated()
    def __init__(
        self,
        base_cell: RecurrentCell,
        zoneout_outputs: float = 0.0,
        zoneout_states: float = 0.0,
    ):
        assert not isinstance(base_cell, BidirectionalCell), (
            "BidirectionalCell doesn't support zoneout since it doesn't support step. "
            "Please add RNNZoneoutCell to the cells underneath instead."
        )
        assert (
            not isinstance(base_cell, SequentialRNNCell)
            or not base_cell._bidirectional
        ), (
            "Bidirectional SequentialRNNCell doesn't support zoneout. "
            "Please add RNNZoneoutCell to the cells underneath instead."
        )
        super(RNNZoneoutCell, self).__init__(base_cell)
        self.zoneout_outputs = zoneout_outputs
        self.zoneout_states = zoneout_states
        self._prev_output = None

    def __repr__(self):
        s = "{name}(p_out={zoneout_outputs}, p_state={zoneout_states}, {base_cell})"
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def _alias(self):
        return "rnnzoneout"

    def reset(self):
        super(RNNZoneoutCell, self).reset()
        self._prev_output = None

    def hybrid_forward(
        self, F, inputs: Tensor, states: Tensor
    ) -> Tuple[Tensor, Tensor]:
        cell, p_outputs, p_states = (
            self.base_cell,
            self.zoneout_outputs,
            self.zoneout_states,
        )
        next_output, next_states = cell(inputs, states)
        mask = lambda p, like: F.Dropout(F.ones_like(like), p=p)

        prev_output = self._prev_output
        if prev_output is None:
            prev_output = F.zeros_like(next_output)

        output_mask = mask(p_outputs, next_output)
        output = (
            F.where(output_mask, next_output, prev_output)
            if p_outputs != 0.0
            else next_output
        )

        # only for RNN, the first element of states is output
        # use the same mask as output, instead of simply copy output to the first element
        # in case that the base cell is ResidualCell
        new_states = [
            F.where(output_mask, next_states[0], states[0])
            if p_outputs != 0.0
            else next_states[0]
        ]
        new_states.extend(
            [
                F.where(mask(p_states, new_s), new_s, old_s)
                for new_s, old_s in zip(next_states[1:], states[1:])
            ]
            if p_states != 0.0
            else next_states[1:]
        )

        self._prev_output = output

        return output, new_states
