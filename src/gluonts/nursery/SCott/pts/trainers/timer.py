# -*- coding: utf-8 -*-
import time
from contextlib import contextmanager
from io import StringIO

import numpy as np
import torch


class Timer:
    """
    Timer for PyTorch code
    Comes in the form of a contextmanager:

    Example:
    >>> timer = Timer()
    ... for i in range(10):
    ...     with timer("expensive operation"):
    ...         x = torch.randn(100)
    ... print(timer.summary())
    """

    def __init__(self, verbosity_level=1, skip_first=True, on_cuda=True):
        self.verbosity_level = verbosity_level
        # self.log_fn = log_fn if log_fn is not None else self._default_log_fn
        self.skip_first = skip_first
        self.cuda_available = torch.cuda.is_available() and on_cuda

        self.reset()

    def reset(self):
        """Reset the timer"""
        self.totals = {}  # Total time per label
        self.first_time = {}  # First occurrence of a label (start time)
        self.last_time = {}  # Last occurence of a label (end time)
        self.call_counts = {}  # Number of times a label occurred

    @contextmanager
    def __call__(self, label, epoch=-1.0, verbosity=1):
        # Don't measure this if the verbosity level is too high
        if verbosity > self.verbosity_level:
            yield
            return

        # Measure the time
        self._cuda_sync()
        start = time.time()
        yield
        self._cuda_sync()
        end = time.time()

        # Update first and last occurrence of this label
        if label not in self.first_time:
            self.first_time[label] = start
        self.last_time[label] = end

        # Update the totals and call counts
        if label not in self.totals and self.skip_first:
            self.totals[label] = 0.0
            del self.first_time[label]
            self.call_counts[label] = 0
        elif label not in self.totals and not self.skip_first:
            self.totals[label] = end - start
            self.call_counts[label] = 1
        else:
            self.totals[label] += end - start
            self.call_counts[label] += 1

    def _cuda_sync(self):
        """Finish all asynchronous GPU computations to get correct timings"""
        if self.cuda_available:
            torch.cuda.synchronize()

    def _default_log_fn(self, _, values, tags):
        label = tags["label"]
        epoch = values["epoch"]
        duration = values["value"]
        print(f"Timer: {label:30s} @ {epoch:4.1f} - {duration:8.5f}s")
