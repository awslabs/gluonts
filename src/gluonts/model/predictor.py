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

# Standard library imports
import itertools
import logging
import multiprocessing as mp
import sys
import traceback
from pathlib import Path
from pydoc import locate
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Callable, Dict, Iterator, List, Optional

# Third-party imports
import mxnet as mx
import numpy as np

# First-party imports
from gluonts.core.component import (
    DType,
    equals,
    from_hyperparameters,
    validated,
)
from gluonts.core.serde import dump_json, fqname_for, load_json
from gluonts.dataset.common import DataEntry, Dataset, ListDataset
from gluonts.dataset.loader import DataBatch, InferenceDataLoader
from gluonts.model.forecast import Forecast, SampleForecast
from gluonts.support.util import (
    export_repr_block,
    export_symb_block,
    get_hybrid_forward_input_names,
    hybrid_block_to_symbol_block,
    import_repr_block,
    import_symb_block,
)
from gluonts.transform import Transformation

if TYPE_CHECKING:  # avoid circular import
    from gluonts.model.estimator import Estimator  # noqa


class Predictor:
    """
    Abstract class representing predictor objects.

    Parameters
    ----------
    prediction_length
        Prediction horizon.
    freq
        Frequency of the predicted data.
    """

    def __init__(self, prediction_length: int, freq: str) -> None:
        assert (
            prediction_length > 0
        ), "The value of `prediction_length` should be > 0"

        self.prediction_length = prediction_length
        self.freq = freq

    def predict(self, dataset: Dataset, **kwargs) -> Iterator[Forecast]:
        """
        Compute forecasts for the time series in the provided dataset.
        This method is not implemented in this abstract class; please
        use one of the subclasses.

        Parameters
        ----------
        dataset
            The dataset containing the time series to predict.

        Returns
        -------
        Iterator[Forecast]
            Iterator over the forecasts, in the same order as the dataset
            iterable was provided.
        """
        raise NotImplementedError

    def serialize(self, path: Path) -> None:
        try:
            # serialize Predictor type
            with (path / 'type.txt').open('w') as fp:
                fp.write(fqname_for(self.__class__))
        except Exception as e:
            raise IOError(
                f'Cannot serialize {fqname_for(self.__class__)} in {path}'
            ) from e

    @classmethod
    def deserialize(cls, path: Path):
        try:
            # deserialize Predictor type
            with (path / 'type.txt').open('r') as fp:
                tpe = locate(fp.readline())
        except Exception as e:
            raise IOError(
                f'Cannot deserialize {fqname_for(cls)} in {path}'
            ) from e

        # ensure that predictor_cls is a subtype of Predictor
        if not issubclass(tpe, Predictor):
            raise IOError(
                f'Class {fqname_for(tpe)} is not '
                f'a subclass of {fqname_for(Predictor)}'
            )

        # call deserialize() for the concrete Predictor type
        return tpe.deserialize(path)

    @classmethod
    def from_hyperparameters(cls, **hyperparameters):
        return from_hyperparameters(cls, **hyperparameters)


class RepresentablePredictor(Predictor):
    """
    An abstract predictor that can be subclassed by models that are not based
    on Gluon. Subclasses should have @validated() constructors.
    (De)serialization and value equality are all implemented on top of the
    @validated() logic.

    Parameters
    ----------
    prediction_length
        Prediction horizon.
    freq
        Frequency of the predicted data.
    """

    @validated()
    def __init__(self, prediction_length: int, freq: str) -> None:
        super().__init__(prediction_length, freq)

    def predict(self, dataset: Dataset, **kwargs) -> Iterator[Forecast]:
        raise NotImplementedError

    def __eq__(self, that):
        """
        Two RepresentablePredictor instances are considered equal if they
        have the same constructor arguments.
        """
        return equals(self, that)

    def serialize(self, path: Path) -> None:
        # call Predictor.serialize() in order to serialize the class name
        super().serialize(path)
        try:
            with (path / 'predictor.json').open('w') as fp:
                print(dump_json(self), file=fp)
        except Exception as e:
            raise IOError(
                f'Cannot serialize {fqname_for(self.__class__)}'
            ) from e

    @classmethod
    def deserialize(cls, path: Path):
        try:
            with (path / 'predictor.json').open('r') as fp:
                return load_json(fp.read())
        except Exception as e:
            raise IOError(f'Cannot deserialize {fqname_for(cls)}') from e


class GluonPredictor(Predictor):
    """
    Base predictor type for Gluon-based models.

    Parameters
    ----------
    input_names
        Input tensor names for the graph
    prediction_net
        Network that will be called for prediction
    batch_size
        Number of time series to predict in a single batch
    prediction_length
        Number of time steps to predict
    freq
        Frequency of the input data
    input_transform
        Input transformation pipeline
    output_transform
        Output transformation
    ctx
        MXNet context to use for computation
    forecast_cls_name
        Class name of the forecast type that will be generated
    forecast_kwargs
        A dictionary that will be passed as kwargs when instantiating the
        forecast object
    """

    BlockType = mx.gluon.Block

    def __init__(
        self,
        input_names: List[str],
        prediction_net: BlockType,
        batch_size: int,
        prediction_length: int,
        freq: str,
        ctx: mx.Context,
        input_transform: Transformation,
        output_transform: Optional[
            Callable[[DataEntry, np.ndarray], np.ndarray]
        ] = None,
        float_type: DType = np.float32,
        forecast_cls_name: str = 'SampleForecast',
        forecast_kwargs: Optional[Dict] = None,
    ) -> None:
        super().__init__(prediction_length, freq)

        forecast_dict = {c.__name__: c for c in Forecast.__subclasses__()}
        assert forecast_cls_name in forecast_dict

        self.input_names = input_names
        self.prediction_net = prediction_net
        self.batch_size = batch_size
        self.input_transform = input_transform
        self.output_transform = output_transform
        self.ctx = ctx
        self.float_type = float_type
        self.forecast_cls_name = forecast_cls_name
        self._forecast_cls = forecast_dict[forecast_cls_name]
        self.forecast_kwargs = forecast_kwargs if forecast_kwargs else {}

    def hybridize(self, batch: DataBatch) -> None:
        """
        Hybridizes the underlying prediction network.

        Parameters
        ----------
        batch
            A batch of data to use for the required forward pass after the
            `hybridize()` call.
        """
        self.prediction_net.hybridize(active=True)
        self.prediction_net(*[batch[k] for k in self.input_names])

    def as_symbol_block_predictor(
        self, batch: DataBatch
    ) -> 'SymbolBlockPredictor':
        """
        Returns a variant of the current :class:`GluonPredictor` backed
        by a Gluon `SymbolBlock`. If the current predictor is already a
        :class:`SymbolBlockPredictor`, it just returns itself.

        Parameters
        ----------
        batch
            A batch of data to use for the required forward pass after the
            `hybridize()` call of the underlying network.

        Returns
        -------
        SymbolBlockPredictor
            A predictor derived from the current one backed by a `SymbolBlock`.
        """
        raise NotImplementedError

    def predict(
        self, dataset: Dataset, num_eval_samples: Optional[int] = None
    ) -> Iterator[Forecast]:
        inference_data_loader = InferenceDataLoader(
            dataset,
            self.input_transform,
            self.batch_size,
            ctx=self.ctx,
            float_type=self.float_type,
        )
        for batch in inference_data_loader:
            inputs = [batch[k] for k in self.input_names]
            outputs = self.prediction_net(*inputs).asnumpy()
            if self.output_transform is not None:
                outputs = self.output_transform(batch, outputs)
            if num_eval_samples and not self._forecast_cls == SampleForecast:
                logging.info(
                    'Forecast is not sample based. Ignoring parameter `num_eval_samples` from predict method.'
                )
            if num_eval_samples and self._forecast_cls == SampleForecast:
                num_collected_samples = outputs[0].shape[0]
                collected_samples = [outputs]
                while num_collected_samples < num_eval_samples:
                    outputs = self.prediction_net(*inputs).asnumpy()
                    if self.output_transform is not None:
                        outputs = self.output_transform(batch, outputs)
                    collected_samples.append(outputs)
                    num_collected_samples += outputs[0].shape[0]
                outputs = [
                    np.concatenate(s)[:num_eval_samples]
                    for s in zip(*collected_samples)
                ]
                assert len(outputs[0]) == num_eval_samples
            assert len(batch['forecast_start']) == len(outputs)
            for i, output in enumerate(outputs):
                yield self._forecast_cls(
                    output,
                    start_date=batch['forecast_start'][i],
                    freq=self.freq,
                    item_id=batch['item_id'][i]
                    if 'item_id' in batch
                    else None,
                    info=batch['info'][i] if 'info' in batch else None,
                    **self.forecast_kwargs,
                )

    def __eq__(self, that):
        if type(self) != type(that):
            return False

        # TODO: also consider equality of the pipelines
        # if not equals(self.input_transform, that.input_transform):
        #    return False

        return equals(
            self.prediction_net.collect_params(),
            that.prediction_net.collect_params(),
        )

    def serialize(self, path: Path) -> None:
        # call Predictor.serialize() in order to serialize the class name
        super().serialize(path)

        # serialize every GluonPredictor-specific parameters
        try:
            # serialize the prediction network
            self.serialize_prediction_net(path)

            # serialize transformation chain
            with (path / 'input_transform.json').open('w') as fp:
                print(dump_json(self.input_transform), file=fp)

            # FIXME: also needs to serialize the output_transform

            # serialize all remaining constructor parameters
            with (path / 'parameters.json').open('w') as fp:
                parameters = dict(
                    batch_size=self.batch_size,
                    prediction_length=self.prediction_length,
                    freq=self.freq,
                    ctx=self.ctx,
                    float_type=self.float_type,
                    forecast_cls_name=self.forecast_cls_name,
                    forecast_kwargs=self.forecast_kwargs,
                    input_names=self.input_names,
                )
                print(dump_json(parameters), file=fp)
        except Exception as e:
            raise IOError(
                f'Cannot serialize {fqname_for(self.__class__)}'
            ) from e

    def serialize_prediction_net(self, path: Path) -> None:
        raise NotImplementedError()


class SymbolBlockPredictor(GluonPredictor):
    """
    A predictor which serializes the network structure as an MXNet symbolic
    graph. Should be used for models deployed in production in order to
    ensure forward-compatibility as GluonTS models evolve.

    Used by the training shell if training is invoked with a hyperparameter
    `use_symbol_block_predictor = True`.
    """

    BlockType = mx.gluon.SymbolBlock

    def as_symbol_block_predictor(
        self, batch: DataBatch
    ) -> 'SymbolBlockPredictor':
        return self

    def serialize(self, path: Path) -> None:
        logging.warning(
            'Serializing RepresentableBlockPredictor instances does not save '
            'the prediction network structure in a backwards-compatible '
            'manner. Be careful not to use this method in production.'
        )
        super().serialize(path)

    def serialize_prediction_net(self, path: Path) -> None:
        export_symb_block(self.prediction_net, path, 'prediction_net')

    @classmethod
    def deserialize(cls, path: Path):
        try:
            # deserialize constructor parameters
            with (path / 'parameters.json').open('r') as fp:
                parameters = load_json(fp.read())

            # deserialize transformation chain
            with (path / 'input_transform.json').open('r') as fp:
                transform = load_json(fp.read())

            # deserialize prediction network
            num_inputs = len(parameters['input_names'])
            prediction_net = import_symb_block(
                num_inputs, path, 'prediction_net'
            )

            return SymbolBlockPredictor(
                input_transform=transform,
                prediction_net=prediction_net,
                **parameters,
            )
        except Exception as e:
            raise IOError(f'Cannot deserialize {fqname_for(cls)}') from e


class RepresentableBlockPredictor(GluonPredictor):
    """
    A predictor which serializes the network structure using the
    JSON-serialization methods located in `gluonts.core.serde`. Use the following
    logic to create a `RepresentableBlockPredictor` from a trained prediction
    network.

    >>> def create_representable_block_predictor(
    ...        prediction_network: mx.gluon.HybridBlock,
    ...        **kwargs
    ... ) -> RepresentableBlockPredictor:
    ...    return RepresentableBlockPredictor(
    ...        prediction_net=prediction_network,
    ...        **kwargs
    ...    )
    """

    BlockType = mx.gluon.HybridBlock

    def __init__(
        self,
        prediction_net: BlockType,
        batch_size: int,
        prediction_length: int,
        freq: str,
        ctx: mx.Context,
        input_transform: Transformation,
        output_transform: Optional[
            Callable[[DataEntry, np.ndarray], np.ndarray]
        ] = None,
        float_type: DType = np.float32,
        forecast_cls_name: str = 'SampleForecast',
        forecast_kwargs: Optional[Dict] = None,
    ) -> None:
        super().__init__(
            input_names=get_hybrid_forward_input_names(prediction_net),
            prediction_net=prediction_net,
            batch_size=batch_size,
            prediction_length=prediction_length,
            freq=freq,
            ctx=ctx,
            input_transform=input_transform,
            output_transform=output_transform,
            float_type=float_type,
            forecast_cls_name=forecast_cls_name,
            forecast_kwargs=forecast_kwargs,
        )

    def as_symbol_block_predictor(
        self, batch: DataBatch
    ) -> SymbolBlockPredictor:
        symbol_block_net = hybrid_block_to_symbol_block(
            hb=self.prediction_net,
            data_batch=[batch[k] for k in self.input_names],
        )

        return SymbolBlockPredictor(
            input_names=self.input_names,
            prediction_net=symbol_block_net,
            batch_size=self.batch_size,
            prediction_length=self.prediction_length,
            freq=self.freq,
            ctx=self.ctx,
            input_transform=self.input_transform,
            output_transform=self.output_transform,
            float_type=self.float_type,
            forecast_cls_name=self.forecast_cls_name,
            forecast_kwargs=self.forecast_kwargs,
        )

    def serialize_prediction_net(self, path: Path) -> None:
        export_repr_block(self.prediction_net, path, 'prediction_net')

    @classmethod
    def deserialize(cls, path: Path):
        try:
            # deserialize constructor parameters
            with (path / 'parameters.json').open('r') as fp:
                parameters = load_json(fp.read())

            # deserialize transformation chain
            with (path / 'input_transform.json').open('r') as fp:
                transform = load_json(fp.read())

            # deserialize prediction network
            prediction_net = import_repr_block(path, 'prediction_net')

            # input_names is derived from the prediction_net
            if 'input_names' in parameters:
                del parameters['input_names']

            return RepresentableBlockPredictor(
                input_transform=transform,
                prediction_net=prediction_net,
                **parameters,
            )
        except Exception as e:
            raise IOError(f'Cannot deserialize {fqname_for(cls)}') from e


class WorkerError:
    def __init__(self, msg):
        self.msg = msg


def _worker_loop(
    predictor_path: Path,
    input_queue: mp.Queue,
    output_queue: mp.Queue,
    worker_id,
    **kwargs,
):
    """
    Worker loop for multiprocessing Predictor.
    Loads the predictor serialized in predictor_path
    reads inputs from input_queue and writes forecasts to output_queue
    """

    predictor = Predictor.deserialize(predictor_path)
    while True:
        idx, data_chunk = input_queue.get()
        if idx is None:
            output_queue.put((None, None, None))
            break
        try:
            result = list(predictor.predict(data_chunk, **kwargs))
        except Exception:
            we = WorkerError(
                "".join(traceback.format_exception(*sys.exc_info()))
            )
            output_queue.put((we, None, None))
            break
        output_queue.put((idx, worker_id, result))


class ParallelizedPredictor(Predictor):
    """
    Runs multiple instances (workers) of a predictor in parallel.

    Exceptions are propagated from the workers.

    Note: That there is currently an issue with tqdm that will cause things
    to hang if the ParallelizedPredictor is used with tqdm and an exception
    occurs during prediction.

    https://github.com/tqdm/tqdm/issues/548

    Parameters
    ----------
    base_predictor
        A representable predictor that will be used
    num_workers
        Number of workers (processes) to use. If set to
        None, one worker per CPU will be used.
    chunk_size
        Number of items to pass per call
    """

    def __init__(
        self,
        base_predictor: Predictor,
        num_workers: Optional[int] = None,
        chunk_size=1,
    ) -> None:
        super().__init__(base_predictor.prediction_length, base_predictor.freq)

        self._base_predictor = base_predictor
        self._num_workers = (
            num_workers if num_workers is not None else mp.cpu_count()
        )
        self._chunk_size = chunk_size
        self._num_running_workers = 0
        self._input_queues = []
        self._output_queue = None

    def _grouper(self, iterable, n):
        iterator = iter(iterable)
        group = tuple(itertools.islice(iterator, n))
        while group:
            yield group
            group = tuple(itertools.islice(iterator, n))

    def terminate(self):
        for q in self._input_queues:
            q.put((None, None))
        for w in self._workers:
            w.terminate()
        for i, w in enumerate(self._workers):
            w.join()

    def predict(self, dataset: Dataset, **kwargs) -> Iterator[Forecast]:
        with TemporaryDirectory() as tempdir:
            predictor_path = Path(tempdir)
            self._base_predictor.serialize(predictor_path)

            # TODO: Consider using shared memory for the data transfer.

            self._input_queues = [mp.Queue() for _ in range(self._num_workers)]
            self._output_queue = mp.Queue()

            workers = []
            for worker_id, in_q in enumerate(self._input_queues):
                worker = mp.Process(
                    target=_worker_loop,
                    args=(predictor_path, in_q, self._output_queue, worker_id),
                    kwargs=kwargs,
                )

                worker.daemon = True
                worker.start()
                workers.append(worker)
                self._num_running_workers += 1

            self._workers = workers

            chunked_data = self._grouper(dataset, self._chunk_size)

            self._send_idx = 0
            self._next_idx = 0

            self._data_buffer = {}

            worker_ids = list(range(self._num_workers))

            def receive():
                idx, worker_id, result = self._output_queue.get()
                if isinstance(idx, WorkerError):
                    self._num_running_workers -= 1
                    self.terminate()
                    raise Exception(idx.msg)
                if idx is not None:
                    self._data_buffer[idx] = result
                return idx, worker_id, result

            def get_next_from_buffer():
                while self._next_idx in self._data_buffer:
                    result_batch = self._data_buffer.pop(self._next_idx)
                    self._next_idx += 1
                    for result in result_batch:
                        yield result

            def send(worker_id, chunk):
                q = self._input_queues[worker_id]
                q.put((self._send_idx, chunk))
                self._send_idx += 1

            try:
                # prime the queues
                for wid in worker_ids:
                    chunk = next(chunked_data)
                    send(wid, chunk)

                while True:
                    idx, wid, result = receive()
                    for res in get_next_from_buffer():
                        yield res
                    chunk = next(chunked_data)
                    send(wid, chunk)
            except StopIteration:
                # signal workers end of data
                for q in self._input_queues:
                    q.put((None, None))

            # collect any outstanding results
            while self._num_running_workers > 0:
                idx, worker_id, result = receive()
                if idx is None:
                    self._num_running_workers -= 1
                    continue
                for res in get_next_from_buffer():
                    yield res
            assert len(self._data_buffer) == 0
            assert self._send_idx == self._next_idx


class Localizer(Predictor):
    """
    A Predictor that uses an estimator to train a local model per time series and
    immediatly calls this to predict.

    Parameters
    ----------
    estimator
        The estimator object to train on each dataset entry at prediction time.
    """

    def __init__(self, estimator: 'Estimator'):
        super().__init__(estimator.prediction_length, estimator.freq)
        self.estimator = estimator

    def predict(self, dataset: Dataset, **kwargs) -> Iterator[Forecast]:
        logger = logging.getLogger(__name__)
        for i, ts in enumerate(dataset, start=1):
            logger.info(f'training for time series {i} / {len(dataset)}')
            local_ds = ListDataset([ts], freq=self.freq)
            trained_pred = self.estimator.train(local_ds)
            logger.info(f'predicting for time series {i} / {len(dataset)}')
            predictions = trained_pred.predict(local_ds, **kwargs)
            for pred in predictions:
                yield pred
