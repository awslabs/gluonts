# Standard library imports
import logging
import os
import tempfile
import time
import uuid
from typing import Any, List, NamedTuple, Optional, Union

# Third-party imports
import mxnet as mx
import mxnet.autograd as autograd
import mxnet.gluon.nn as nn
import numpy as np

# First-party imports
from gluonts.core.component import has_gpu_support, validated
from gluonts.core.exception import GluonTSDataError
from gluonts.dataset.loader import TrainDataLoader
from gluonts.support.util import HybridContext
from gluonts.gluonts_tqdm import tqdm

# Relative imports
from . import learning_rate_scheduler as lrs

logger = logging.getLogger('trainer')

MODEL_ARTIFACT_FILE_NAME = 'model'
STATE_ARTIFACT_FILE_NAME = 'state'

# make the IDE happy: mx.py does not explicitly import autograd
mx.autograd = autograd


def check_loss_finite(val: float) -> None:
    if not np.isfinite(val):
        raise GluonTSDataError(
            "Encountered invalid loss value! Try reducing the learning rate "
            "or try a different likelihood."
        )


def loss_value(loss: mx.metric.Loss) -> float:
    return loss.get_name_value()[0][1]


class BestEpochInfo(NamedTuple):
    params_path: str
    epoch_no: int
    metric_value: float


class Trainer:
    @validated()
    def __init__(
        self,
        ctx: Optional[mx.Context] = None,
        epochs: int = 1,
        batch_size: int = 32,
        num_batches_per_epoch: int = 100,
        learning_rate: float = 1e-3,
        learning_rate_decay_factor: float = 0.5,
        patience: int = 10,
        minimum_learning_rate: float = 5e-5,
        clip_gradient: float = 10.0,
        weight_decay: float = 1e-8,
        init: Union[str, mx.initializer.Initializer] = 'xavier',
        train_log_interval: int = 50,
        hybridize: bool = True,
    ) -> None:

        assert (
            0 < train_log_interval < float('inf')
        ), "The value of `train_log_interval` should be > 0"
        assert (
            0 <= epochs < float('inf')
        ), "The value of `epochs` should be >= 0"
        assert 0 < batch_size, "The value of `batch_size` should be > 0"
        assert (
            0 < num_batches_per_epoch
        ), "The value of `num_batches_per_epoch` should be > 0"
        assert (
            0 < learning_rate < float('inf')
        ), "The value of `learning_rate` should be > 0"
        assert (
            0 <= learning_rate_decay_factor < 1
        ), "The value of `learning_rate_decay_factor` should be in the [0, 1) range"
        assert 0 <= patience, "The value of `patience` should be >= 0"
        assert (
            0 <= minimum_learning_rate
        ), "The value of `minimum_learning_rate` should be >= 0"
        assert 0 < clip_gradient, "The value of `clip_gradient` should be > 0"
        assert 0 <= weight_decay, "The value of `weight_decay` should be => 0"
        assert (
            1 <= train_log_interval
        ), "The value of `train_log_interval` should be >= 1"

        self.epochs = epochs
        self.batch_size = batch_size
        self.num_batches_per_epoch = num_batches_per_epoch
        self.learning_rate = learning_rate
        self.learning_rate_decay_factor = learning_rate_decay_factor
        self.patience = patience
        self.minimum_learning_rate = minimum_learning_rate
        self.clip_gradient = clip_gradient
        self.weight_decay = weight_decay
        self.init = init
        self.train_log_interval = train_log_interval
        self.hybridize = hybridize
        self.ctx = (
            ctx
            if ctx is not None
            else mx.Context('gpu')
            if has_gpu_support()
            else mx.Context('cpu')
        )
        self.halt = False

    def set_halt(self, signum: int, stack_frame: Any) -> None:
        logging.info("Received signal: {}".format(signum))
        self.halt = True

    def count_model_params(self, net: nn.HybridBlock) -> int:
        params = net.collect_params()
        num_params = 0
        for p in params:
            v = params[p]
            num_params += np.prod(v.shape)
        return num_params

    def __call__(
        self,
        net: nn.HybridBlock,
        input_names: List[str],
        train_iter: TrainDataLoader,
    ) -> None:  # TODO: we may want to return some training information here
        self.halt = False

        with tempfile.TemporaryDirectory(
            prefix='gluonts-trainer-temp-', dir='/tmp'
        ) as gluonts_temp:

            def base_path() -> str:
                return os.path.join(
                    gluonts_temp,
                    "{}_{}".format(STATE_ARTIFACT_FILE_NAME, uuid.uuid4()),
                )

            logging.info("Start model training")

            net.initialize(ctx=self.ctx, init=self.init)

            with HybridContext(
                net=net,
                hybridize=self.hybridize,
                static_alloc=True,
                static_shape=True,
            ):
                net_name = type(net).__name__
                num_model_param = self.count_model_params(net)
                logging.info(
                    f"Number of parameters in {net_name}: {num_model_param}"
                )

                batch_size = train_iter.batch_size
                epoch_loss = mx.metric.Loss()

                best_epoch_info = BestEpochInfo(
                    params_path='%s-%s.params' % (base_path(), 'init'),
                    epoch_no=-1,
                    metric_value=np.Inf,
                )

                lr_scheduler = lrs.MetricAttentiveScheduler(
                    objective="min",
                    patience=self.patience,
                    decay_factor=self.learning_rate_decay_factor,
                    min_lr=self.minimum_learning_rate,
                )

                optimizer = mx.optimizer.Adam(
                    learning_rate=self.learning_rate,
                    lr_scheduler=lr_scheduler,
                    wd=self.weight_decay,
                    clip_gradient=self.clip_gradient,
                )

                trainer = mx.gluon.Trainer(
                    net.collect_params(),
                    optimizer=optimizer,
                    kvstore='device',  # FIXME: initialize properly
                )

                for epoch_no in range(self.epochs):
                    if self.halt:
                        logging.info(
                            f"Epoch[{epoch_no}] Interrupting training"
                        )
                        break

                    curr_lr = trainer.learning_rate
                    logging.info(
                        f"Epoch[{epoch_no}] Learning rate is {curr_lr}"
                    )

                    # mark epoch start time
                    tic = time.time()

                    epoch_loss.reset()

                    with tqdm(train_iter) as it:
                        for batch_no, data_entry in enumerate(it, start=1):
                            if self.halt:
                                break

                            inputs = [data_entry[k] for k in input_names]

                            with mx.autograd.record():
                                output = net(*inputs)

                                # network can returns several outputs, the first being always the loss
                                # when having multiple outputs, the forward returns a list in the case of hybrid and a
                                # tuple otherwise
                                # we may wrap network outputs in the future to avoid this type check
                                if isinstance(output, (list, tuple)):
                                    loss = output[0]
                                else:
                                    loss = output

                            loss.backward()
                            trainer.step(batch_size)

                            epoch_loss.update(None, preds=loss)
                            it.set_postfix(
                                ordered_dict={
                                    'avg_epoch_loss': loss_value(epoch_loss)
                                },
                                refresh=False,
                            )

                    # mark epoch end time and log time cost of current epoch
                    toc = time.time()
                    logging.info(
                        'Epoch[%d] Elapsed time %.3f seconds',
                        epoch_no,
                        (toc - tic),
                    )

                    # check and log epoch loss
                    check_loss_finite(loss_value(epoch_loss))
                    logging.info(
                        'Epoch[%d] Evaluation metric \'%s\'=%f',
                        epoch_no,
                        "epoch_loss",
                        loss_value(epoch_loss),
                    )

                    lr_scheduler.step(loss_value(epoch_loss))

                    if loss_value(epoch_loss) < best_epoch_info.metric_value:
                        best_epoch_info = BestEpochInfo(
                            params_path='%s-%04d.params'
                            % (base_path(), epoch_no),
                            epoch_no=epoch_no,
                            metric_value=loss_value(epoch_loss),
                        )
                        net.save_parameters(
                            best_epoch_info.params_path
                        )  # TODO: handle possible exception

                    if not trainer.learning_rate == curr_lr:
                        logging.info(
                            f"Loading parameters from best epoch "
                            f"({best_epoch_info.epoch_no})"
                        )
                        net.load_parameters(
                            best_epoch_info.params_path, self.ctx
                        )

                logging.info(
                    f"Loading parameters from best epoch "
                    f"({best_epoch_info.epoch_no})"
                )
                net.load_parameters(best_epoch_info.params_path, self.ctx)

                logging.info(
                    f"Final loss: {best_epoch_info.metric_value} "
                    f"(occurred at epoch {best_epoch_info.epoch_no})"
                )

                # save net parameters
                net.save_parameters(best_epoch_info.params_path)

                logging.getLogger().info("End model training")
