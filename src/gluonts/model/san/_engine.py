from typing import Optional, List, Callable, Iterator
import os
import logging
import tempfile
import inspect
import uuid

import numpy as np
import mxnet as mx
from mxnet.gluon import nn, HybridBlock, Trainer

from gluonts.mx.trainer import Trainer as BaseTrainer
from gluonts.mx.trainer.learning_rate_scheduler import MetricAttentiveScheduler
from gluonts.mx.trainer.model_averaging import (
    AveragingStrategy,
    SelectNBestMean,
    save_epoch_info,
)
from gluonts.model.forecast import QuantileForecast
from gluonts.model.forecast_generator import (
    QuantileForecastGenerator as BaseForecastGenerator,
)
from gluonts.dataset.loader import (
    DataLoader,
    TrainDataLoader,
    ValidationDataLoader,
    InferenceDataLoader,
)
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.common import DataEntry
from gluonts.gluonts_tqdm import tqdm
from gluonts.support.util import HybridContext

logger = logging.getLogger("gluonts").getChild("trainer")
MODEL_ARTIFACT_FILE_NAME = "model"
STATE_ARTIFACT_FILE_NAME = "state"
LOG_CACHE = set([])
OutputTransform = Callable[[DataEntry, np.ndarray], np.ndarray]


class Trainer(BaseTrainer):
    def __call__(
        self,
        net: nn.HybridBlock,
        input_names: List[str],
        train_iter: TrainDataLoader,
        validation_iter: Optional[ValidationDataLoader] = None,
    ) -> None:  # TODO: we may want to return some training information here
        is_validation_available = validation_iter is not None
        self.halt = False

        with tempfile.TemporaryDirectory(
            prefix="gluonts-trainer-temp-"
        ) as temp:

            def base_path() -> str:
                return os.path.join(
                    temp,
                    "{}_{}".format(STATE_ARTIFACT_FILE_NAME, uuid.uuid4()),
                )

            def loss_value(loss: mx.metric.Loss) -> float:
                return loss.get_name_value()[0][1]

            logger.info("Start model training")

            net.initialize(ctx=self.ctx, init=self.init)

            with HybridContext(
                net=net,
                hybridize=self.hybridize,
                static_alloc=True,
                static_shape=True,
            ):
                batch_size = train_iter.batch_size

                best_epoch_info = {
                    "params_path": f"{base_path()}-init.params",
                    "epoch_no": -1,
                    "score": np.Inf,
                }

                lr_scheduler = MetricAttentiveScheduler(
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

                trainer = Trainer(
                    net.collect_params(),
                    optimizer=optimizer,
                    kvstore="device",  # FIXME: initialize properly
                )

                def loop(
                    epoch_no: int,
                    batch_iter: DataLoader,
                    is_training: bool = True,
                ) -> mx.metric.Loss:
                    tic = time.time()
                    epoch_loss = mx.metric.Loss()
                    with tqdm(batch_iter) as it:
                        for batch_no, data_entry in enumerate(it, start=1):
                            if self.halt:
                                break
                            args = inspect.signature(
                                net.hybrid_forward
                            ).parameters
                            inputs = []
                            for n, (name, arg) in enumerate(args.items()):
                                if n == 0:
                                    if name == "F":
                                        continue
                                    else:
                                        raise RuntimeError(
                                            f"Expected first argument of HybridBlock to be `F`, "
                                            f"but found `{name}`"
                                        )
                                if name in data_entry:
                                    inputs.append(data_entry[name])
                                elif not (arg.default is inspect._empty):
                                    inputs.append(arg.default)
                                else:
                                    raise RuntimeError(
                                        f"The value of argument `{name}` of HybridBlock is not provided, "
                                        f"and no default value is set."
                                    )
                            with mx.autograd.record():
                                loss = net(*inputs)

                            if is_training:
                                loss.backward()
                                trainer.step(batch_size)

                            epoch_loss.update(None, preds=loss)
                            lv = loss_value(epoch_loss)

                            if not np.isfinite(lv):
                                logger.warning(
                                    "Epoch[%d] gave nan loss", epoch_no
                                )
                                return epoch_loss

                            it.set_postfix(
                                ordered_dict={
                                    "epoch": f"{epoch_no + 1}/{self.epochs}",
                                    f"{'' if is_training else 'val_'}avg_epoch_loss": lv,
                                },
                                refresh=False,
                            )
                            # print out parameters of the network at the first pass
                            if batch_no == 1 and epoch_no == 0:
                                net_name = type(net).__name__
                                num_model_param = self.count_model_params(net)
                                logger.info(
                                    f"Number of parameters in {net_name}: {num_model_param}"
                                )
                    # mark epoch end time and log time cost of current epoch
                    toc = time.time()
                    logger.info(
                        "Epoch[%d] Elapsed time %.3f seconds",
                        epoch_no,
                        (toc - tic),
                    )

                    logger.info(
                        "Epoch[%d] Evaluation metric '%s'=%f",
                        epoch_no,
                        f"{'' if is_training else 'val_'}epoch_loss",
                        lv,
                    )
                    return lv

                for epoch_no in range(self.epochs):
                    if self.halt:
                        logger.info(f"Epoch[{epoch_no}] Interrupting training")
                        break

                    curr_lr = trainer.learning_rate
                    logger.info(
                        f"Epoch[{epoch_no}] Learning rate is {curr_lr}"
                    )

                    epoch_loss = loop(epoch_no, train_iter, is_training=True)
                    if is_validation_available:
                        epoch_loss = loop(
                            epoch_no, validation_iter, is_training=False
                        )

                    should_continue = lr_scheduler.step(epoch_loss)
                    if not should_continue:
                        logger.info("Stopping training")
                        break

                    # save model and epoch info
                    bp = base_path()
                    epoch_info = {
                        "params_path": f"{bp}-0000.params",
                        "epoch_no": epoch_no,
                        "score": epoch_loss,
                    }
                    net.save_parameters(
                        epoch_info["params_path"]
                    )  # TODO: handle possible exception
                    save_epoch_info(bp, epoch_info)

                    # update best epoch info - needed for the learning rate scheduler
                    if epoch_loss < best_epoch_info["score"]:
                        best_epoch_info = epoch_info.copy()

                    if not trainer.learning_rate == curr_lr:
                        if best_epoch_info["epoch_no"] == -1:
                            raise GluonTSUserError(
                                "Got NaN in first epoch. Try reducing initial learning rate."
                            )

                        logger.info(
                            f"Loading parameters from best epoch "
                            f"({best_epoch_info['epoch_no']})"
                        )
                        net.load_parameters(
                            best_epoch_info["params_path"], self.ctx
                        )

                logging.info("Computing averaged parameters.")
                averaged_params_path = self.avg_strategy.apply(temp)

                logging.info("Loading averaged parameters.")
                net.load_parameters(averaged_params_path, self.ctx)

                logger.info("End model training")


class QuantileForecastGenerator(BaseForecastGenerator):
    def __call__(
        self,
        inference_data_loader: InferenceDataLoader,
        prediction_net: HybridBlock,
        input_names: List[str],
        freq: str,
        output_transform: Optional[OutputTransform],
        num_samples: Optional[int],
        **kwargs,
    ) -> Iterator[QuantileForecast]:
        def log_once(msg):
            global LOG_CACHE
            if msg not in LOG_CACHE:
                logging.info(msg)
                LOG_CACHE.add(msg)

        for batch in inference_data_loader:
            args = inspect.signature(prediction_net.hybrid_forward).parameters
            inputs = []
            for n, (name, arg) in enumerate(args.items()):
                if n == 0:
                    if name == "F":
                        continue
                    else:
                        raise RuntimeError(
                            f"Expected first argument of HybridBlock to be `F`, "
                            f"but found `{name}`"
                        )
                if name in batch:
                    inputs.append(batch[name])
                elif not (arg.default is inspect._empty):
                    inputs.append(arg.default)
                else:
                    raise RuntimeError(
                        f"The value of argument `{name}` of HybridBlock is not provided, "
                        f"and no default value is set."
                    )
            outputs = prediction_net(*inputs).asnumpy()
            if output_transform is not None:
                outputs = output_transform(batch, outputs)

            if num_samples:
                log_once(
                    "Forecast is not sample based. Ignoring parameter `num_samples` from predict method."
                )

            i = -1
            for i, output in enumerate(outputs):
                yield QuantileForecast(
                    output,
                    start_date=batch["forecast_start"][i],
                    freq=freq,
                    item_id=batch[FieldName.ITEM_ID][i]
                    if FieldName.ITEM_ID in batch
                    else None,
                    info=batch["info"][i] if "info" in batch else None,
                    forecast_keys=self.quantiles,
                )
            assert i + 1 == len(batch["forecast_start"])
