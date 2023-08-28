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


from pathlib import Path
from typing import Dict, Optional, List, Iterator, Tuple

import pandas as pd

from gluonts.core.component import validated
from gluonts.dataset.common import Dataset
from gluonts.dataset.util import forecast_start
from gluonts.model.forecast import Forecast
from gluonts.model.predictor import RepresentablePredictor
from gluonts.time_feature import get_seasonality

# https://stackoverflow.com/questions/25329955/check-if-r-is-installed-from-python
from subprocess import Popen, PIPE

proc = Popen(["which", "R"], stdout=PIPE, stderr=PIPE)
R_IS_INSTALLED = proc.wait() == 0

try:
    import rpy2.robjects.packages as rpackages
    from rpy2 import rinterface, robjects
    from rpy2.rinterface_lib import callbacks
    from rpy2.rinterface_lib.embedded import RRuntimeError
except ImportError as e:
    rpy2_error_message = str(e)
    RPY2_IS_INSTALLED = False
else:
    RPY2_IS_INSTALLED = True

USAGE_MESSAGE = """
The RForecastPredictor is a thin wrapper for calling the R forecast package.
In order to use it you need to install R and rpy2. You also need to install \
specific R packages for univariate and/or hierarchical methods.

For univariate methods, install:
R -e 'install.packages(c("forecast", "nnfor"),\
repos="https://cloud.r-project.org")'

For hierarchical methods, install:
R -e 'install.packages(c("hts"), repos="https://cloud.r-project.org")'
"""


class RBasePredictor(RepresentablePredictor):
    """
    The `RBasePredictor` is a thin wrapper for calling R packages.
    In order to use it you need to install R and rpy2.

    Note that specific R packages need to be installed, depending
    on which wrapper one needs to run.
    See `RForecastPredictor` and `RHierarchicalForecastPredictor` to know
    which packages are needed.

    Parameters
    ----------
    freq
        The granularity of the time series (e.g. '1H')
    prediction_length
        Number of time points to be predicted.
    period
        The period to be used (this is called `frequency` in the R forecast
        package), result to a tentative reasonable default if not specified
        (for instance 24 for hourly freq '1H')
    trunc_length
        Maximum history length to feed to the model (some models become slow
        with very long series).
    r_file_prefix
        Prefix string of the R file(s) where our forecasting wrapper methods
        can be found.
        This is to avoid loading all R files potentially having different
        implementations of the same method, thereby making sure the
        expected R method is in fact used.
    """

    @validated()
    def __init__(
        self,
        freq: str,
        prediction_length: int,
        period: int = None,
        trunc_length: Optional[int] = None,
        save_info: bool = False,
        r_file_prefix: str = "",
    ) -> None:
        super().__init__(prediction_length=prediction_length)

        if not R_IS_INSTALLED:
            raise ImportError("R is not Installed! \n " + USAGE_MESSAGE)

        if not RPY2_IS_INSTALLED:
            raise ImportError(rpy2_error_message + USAGE_MESSAGE)

        self._robjects = robjects
        self._rinterface = rinterface
        self._rcallbacks = callbacks
        self._rinterface.initr()
        self._rpackages = rpackages

        this_dir = Path(__file__).resolve().parent.absolute()
        this_dir = Path(f"{this_dir}".replace("\\", "/"))  # for windows
        r_files = this_dir.rglob(f"{r_file_prefix}*.R")

        for r_file in r_files:
            try:
                robjects.r(f'source("{r_file}")'.replace("\\", "\\\\"))
            except RRuntimeError as er:
                raise RRuntimeError(str(er) + USAGE_MESSAGE) from er

        self._stats_pkg = rpackages.importr("stats")

        self.prediction_length = prediction_length
        self.period = period if period is not None else get_seasonality(freq)
        self.trunc_length = trunc_length
        self.save_info = save_info

    def _get_r_forecast(self, data: Dict) -> Dict:
        """
        Get forecasts from an R method.

        Parameters
        ----------
        data
            Dictionary containing the `target` time series.

        Returns
        -------
        Dictionary
            Forecasts saved in a dictionary.

        """
        raise NotImplementedError()

    def _run_r_forecast(self, data: Dict) -> Tuple[Dict, List]:
        """
        Run an R forecast method.

        Parameters
        ----------
        data
            Dictionary containing the `target` time series.

        Returns
        -------
        Tuple[Dict, List]:

        """
        buf = []

        def save_to_buf(x):
            buf.append(x)

        def dont_save(x):
            pass

        f = save_to_buf if self.save_info else dont_save

        # save output from the R console in buf
        consolewrite_print_backup = self._rcallbacks.consolewrite_print
        consolewrite_warnerror_backup = self._rcallbacks.consolewrite_warnerror

        self._rcallbacks.consolewrite_print = f
        self._rcallbacks.consolewrite_warnerror = f

        forecast_dict = self._get_r_forecast(data=data)

        self._rcallbacks.consolewrite_print = consolewrite_print_backup
        self._rcallbacks.consolewrite_warnerror = consolewrite_warnerror_backup

        return forecast_dict, buf

    def _preprocess_data(self, data: Dict) -> Dict:
        """
        Preprocessing of target time series, e.g., truncating length or
        slicing bottom time series in case of hierarchical forecasting etc.

        Parameters
        ----------
        data
            Dictionary containing target time series.

        Returns
        -------
        Dict

        """
        raise NotImplementedError()

    def _warning_message(self) -> None:
        """
        Prints warning messages (once per whole dataset), e.g., if default
        parameters are overridden.

        Returns
        -------

        """
        return

    def _forecast_dict_to_obj(
        self,
        forecast_dict: Dict,
        forecast_start_date: pd.Timestamp,
        item_id: Optional[str],
        info: Dict,
    ) -> Forecast:
        """
        Returns object of type `gluonts.model.Forecast`.

        Parameters
        ----------
        forecast_dict
            Dictionary containing `samples` or `quantiles`.
        forecast_start_date
            Start date of the forecast.
        item_id
            Item identifier.
        info
            Additional information.

        Returns
        -------
        Forecast
            Sample based or quantile based forecasts.

        """
        raise NotImplementedError()

    def predict(
        self,
        dataset: Dataset,
        **kwargs,
    ) -> Iterator[Forecast]:
        self._warning_message()

        for data in dataset:
            data = self._preprocess_data(data=data)

            forecast_dict, console_output = self._run_r_forecast(data)

            info = (
                {"console_output": "\n".join(console_output)}
                if self.save_info
                else None
            )

            yield self._forecast_dict_to_obj(
                forecast_dict=forecast_dict,
                forecast_start_date=forecast_start(data),
                item_id=data.get("item_id", None),
                info=info,
            )
