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

import collections
import re
import warnings
from typing import DefaultDict, List, Optional, Tuple

from mxnet import Context, gluon, nd
from mxnet.gluon import Block, Parameter
from mxnet.ndarray import NDArray

from gluonts.core.component import validated


class WeightDropParameter(Parameter):
    """A Container holding parameters (weights) of Blocks and performs dropout.
    This code is from:
    https://github.com/dmlc/gluon-nlp/blob/v0.10.x/src/gluonnlp/model/parameter.py

    Parameters
    ----------
    parameter : Parameter
        The parameter which drops out.
    rate : float, default 0.0
        Fraction of the input units to drop. Must be a number between 0 and 1.
        Dropout is not applied if dropout_rate is 0.
    mode : str, default 'training'
        Whether to only turn on dropout during training or
         to also turn on for inference.
        Options are 'training' and 'always'.
    axes : tuple of int, default ()
        Axes on which dropout mask is shared.
    """

    @validated()
    def __init__(
        self,
        parameter: Parameter,
        rate: float = 0.0,
        mode: str = "training",
        axes: tuple = (),
    ) -> None:
        p = parameter
        self._deferred_init = p._deferred_init
        super().__init__(
            name=p.name,
            grad_req=p.grad_req,
            shape=p._shape,
            dtype=p.dtype,
            lr_mult=p.lr_mult,
            wd_mult=p.wd_mult,
            init=p.init,
            allow_deferred_init=p._allow_deferred_init,
            differentiable=p._differentiable,
        )
        self._rate = rate
        self._mode = mode
        self._axes = axes
        self._var = p._var
        self._data = p._data
        self._grad = p._grad
        self._ctx_list = p._ctx_list
        self._ctx_map = p._ctx_map
        self._trainer = p._trainer

    def data(self, ctx: Optional[Context] = None) -> NDArray:
        """Returns a copy of this parameter on one context. Must have been
        initialized on this context before.

        Parameters
        ----------
        ctx : Context
            Desired context.
        Returns
        -------
        NDArray on ctx
        """
        d = self._check_and_get(self._data, ctx)
        if self._rate:
            d = nd.Dropout(d, self._rate, self._mode, self._axes)
        return d

    def __repr__(self) -> str:
        s = (
            "WeightDropParameter {name} (shape={shape},"
            " dtype={dtype}, rate={rate}, mode={mode})"
        )
        return s.format(
            name=self.name,
            shape=self.shape,
            dtype=self.dtype,
            rate=self._rate,
            mode=self._mode,
        )


def apply_weight_drop(
    block: Block,
    local_param_regex: str,
    rate: float,
    axes: Tuple = (),
    weight_dropout_mode: str = "training",
) -> None:
    """Apply weight drop to the parameter of a block.
     The code is from
      https://github.com/dmlc/gluon-nlp/blob/v0.10.x/src/gluonnlp/model/utils.py

    Parameters
    ----------
    block : Block or HybridBlock
        The block whose parameter is to be applied weight-drop.
    local_param_regex : str
        The regex for parameter names used in the self.params.get(), such as 'weight'.
    rate : float
        Fraction of the input units to drop. Must be a number between 0 and 1.
    axes : tuple of int, default ()
        The axes on which dropout mask is shared. If empty, regular dropout is applied.
    weight_drop_mode : {'training', 'always'}, default 'training'
        Whether the weight dropout should be applied only at training time,
         or always be applied.

    Examples
    --------
    >>> net = gluon.rnn.LSTM(10, num_layers=2, bidirectional=True)
    >>> gluonnlp.model.apply_weight_drop(net, r'.*h2h_weight', 0.5)
    >>> net.collect_params()
    lstm0_ (
      Parameter lstm0_l0_i2h_weight (shape=(40, 0), dtype=float32)
      WeightDropParameter lstm0_l0_h2h_weight (shape=(40, 10), dtype=float32, \
rate=0.5, mode=training)
      Parameter lstm0_l0_i2h_bias (shape=(40,), dtype=float32)
      Parameter lstm0_l0_h2h_bias (shape=(40,), dtype=float32)
      Parameter lstm0_r0_i2h_weight (shape=(40, 0), dtype=float32)
      WeightDropParameter lstm0_r0_h2h_weight (shape=(40, 10), dtype=float32, \
rate=0.5, mode=training)
      Parameter lstm0_r0_i2h_bias (shape=(40,), dtype=float32)
      Parameter lstm0_r0_h2h_bias (shape=(40,), dtype=float32)
      Parameter lstm0_l1_i2h_weight (shape=(40, 20), dtype=float32)
      WeightDropParameter lstm0_l1_h2h_weight (shape=(40, 10), dtype=float32, \
rate=0.5, mode=training)
      Parameter lstm0_l1_i2h_bias (shape=(40,), dtype=float32)
      Parameter lstm0_l1_h2h_bias (shape=(40,), dtype=float32)
      Parameter lstm0_r1_i2h_weight (shape=(40, 20), dtype=float32)
      WeightDropParameter lstm0_r1_h2h_weight (shape=(40, 10), dtype=float32, \
rate=0.5, mode=training)
      Parameter lstm0_r1_i2h_bias (shape=(40,), dtype=float32)
      Parameter lstm0_r1_h2h_bias (shape=(40,), dtype=float32)
    )
    >>> ones = mx.nd.ones((3, 4, 5))
    >>> net.initialize()
    >>> with mx.autograd.train_mode():
    ...     net(ones).max().asscalar() != net(ones).max().asscalar()
    True
    """
    if not rate:
        return

    existing_params = _find_params(block, local_param_regex)
    for (
        (local_param_name, param),
        (
            ref_params_list,
            ref_reg_params_list,
        ),
    ) in existing_params.items():
        if isinstance(param, WeightDropParameter):
            continue
        dropped_param = WeightDropParameter(
            param, rate, weight_dropout_mode, axes
        )
        for ref_params in ref_params_list:
            ref_params[param.name] = dropped_param
        for ref_reg_params in ref_reg_params_list:
            ref_reg_params[local_param_name] = dropped_param
            if hasattr(block, local_param_name):
                local_attr = getattr(block, local_param_name)
                if local_attr == param:
                    local_attr = dropped_param
                elif isinstance(local_attr, (list, tuple)):
                    if isinstance(local_attr, tuple):
                        local_attr = list(local_attr)
                    for i, v in enumerate(local_attr):
                        if v == param:
                            local_attr[i] = dropped_param
                elif isinstance(local_attr, dict):
                    for k, v in local_attr:
                        if v == param:
                            local_attr[k] = dropped_param
                else:
                    continue
                if local_attr:
                    super(Block, block).__setattr__(
                        local_param_name, local_attr
                    )


def _find_params(
    block: Block, local_param_regex: str
) -> DefaultDict[Tuple[str, Parameter], Tuple[List, List[Parameter]]]:
    # The code is from
    # https://github.com/dmlc/gluon-nlp/blob/v0.10.x/src/gluonnlp/model/utils.py
    # return {(local_param_name, parameter): (referenced_params_list,
    #                                         referenced_reg_params_list)}

    results: DefaultDict = collections.defaultdict(lambda: ([], []))
    pattern = re.compile(local_param_regex)
    local_param_names = (
        (local_param_name, p)
        for local_param_name, p in block._reg_params.items()
        if pattern.match(local_param_name)
    )

    for local_param_name, p in local_param_names:
        ref_params_list, ref_reg_params_list = results[(local_param_name, p)]
        ref_reg_params_list.append(block._reg_params)

        params = block._params
        while params:
            if p.name in params._params:
                ref_params_list.append(params._params)
            if params._shared:
                params = params._shared
                warnings.warn(
                    "When applying weight drop, target parameter {} "
                    "was found in a shared parameter dict. "
                    "The parameter attribute of the original block "
                    "on which the shared parameter dict was attached "
                    "will not be updated with WeightDropParameter. "
                    "If necessary, please update the attribute manually. "
                    "The likely name of the attribute is "
                    '".{}"'.format(p.name, local_param_name)
                )
            else:
                break

    if block._children:
        if isinstance(block._children, list):
            children = block._children
        elif isinstance(block._children, dict):
            children = list(block._children.values())
        for c in children:
            child_results = _find_params(c, local_param_regex)
            for (
                (child_p_name, child_p),
                (
                    child_pd_list,
                    child_rd_list,
                ),
            ) in child_results.items():
                pd_list, rd_list = results[(child_p_name, child_p)]
                pd_list.extend(child_pd_list)
                rd_list.extend(child_rd_list)

    return results
