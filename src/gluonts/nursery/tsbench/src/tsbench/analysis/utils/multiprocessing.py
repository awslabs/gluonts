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

import math
import multiprocessing as mp
import os
from multiprocessing.pool import ThreadPool
from queue import Empty
from typing import Any, Callable, cast, List, TypeVar
import psutil
from tqdm.auto import tqdm

T = TypeVar("T")
U = TypeVar("U")


_GLOBAL_DATA_CACHE = None


def num_fitting_processes(
    cpus_per_process: float = 2, memory_per_process: float = 16
) -> int:
    """
    Returns the number of processes that can be fitted onto the machine when
    using a particular number of CPUs and a particular amount of memory for
    every process.

    Args:
        cpus_per_process: The number of CPUs that every process requires.
        memory_per_process: The memory in GiB that every process needs.

    Returns:
        The number of processes to use.
    """
    num_processes_cpu = math.floor(
        cast(int, os.cpu_count()) / cpus_per_process
    )

    available_gib = psutil.virtual_memory().total / (1024**3)
    num_processes_memory = math.floor(available_gib / memory_per_process)

    return min(num_processes_cpu, num_processes_memory)


def run_parallel(
    execute: Callable[[T], U], data: List[T], num_processes: int
) -> List[U]:
    """
    Runs a function on multiple processes, parallelizing computations for the
    provided data.

    Args:
        execute: The function to run in each process.
        data: The data items to execute the function for.
        num_processes: The number of processes to parallelize over.

    Returns:
        The outputs of the function calls, ordered in the same way as the data.
    """

    def factory(_i: int) -> mp.Process:
        process = mp.Process(target=_worker, args=(execute, inputs, outputs))
        process.start()
        return process

    # We share data via the global data cache such that we do not need to pass too much data over
    # queues -- forking allows to read the same data from multiple processes without issues.
    global _GLOBAL_DATA_CACHE  # pylint: disable=global-statement
    _GLOBAL_DATA_CACHE = data  # type: ignore

    # Initialize queues and put all items into input queue. Also, put as many "done" items into the
    # queue as we have processes
    inputs = mp.Queue()
    outputs = mp.Queue()
    for i in range(len(data)):
        inputs.put(i)

    # Create the processes in a thread pool to speed up creation
    with ThreadPool(num_processes) as p:
        processes = p.map_async(factory, range(num_processes))

        # Parallelize execution -- keep this inside the with statement so processes keep
        # getting spawned
        result = [None] * len(data)
        with tqdm(total=len(data)) as progress:
            progress.set_postfix({"num_processes": num_processes})
            for _ in range(len(data)):
                i, output = outputs.get()
                result[i] = output
                progress.update()

        for p in processes.get():
            p.kill()

    # cleanup
    inputs.close()
    outputs.close()
    return cast(List[U], result)


def _worker(
    execute: Callable[[Any], Any],
    inputs: mp.Queue,  # type: ignore
    outputs: mp.Queue,  # type: ignore
) -> None:
    while True:
        # Timeout is needed to shutdown workers if no tasks are available anymore.
        try:
            i = inputs.get(timeout=10)
        except Empty:
            return
        data = _GLOBAL_DATA_CACHE[i]  # type: ignore
        output = execute(data)
        outputs.put((i, output))
