USAGE_MESSAGE = """
The RForecastPredictor is a thin wrapper for calling the R forecast package.
In order to use it you need to install R and run

pip install rpy2

R -e 'install.packages(c("forecast", "nnfor"), repos="https://cloud.r-project.org")'
"""


def import_r():
    this_dir = Path(__file__).resolve().parent
    forecast_methods = this_dir / "R" / "forecast_methods.R"

    try:
        import rpy2.rinterface
        import rpy2.robjects.robjects
        import rpy2.robjects.robjects.packages
    except ImportError as e:
        raise ImportError(str(e) + USAGE_MESSAGE) from e

    rpy2.rinterface.initr()

    try:
        robjects.r(f'source("{forecast_methods}")')
    except rpy2.rinterface.RRuntimeError as err:
        raise RRuntimeError(str(err) + USAGE_MESSAGE) from err

    return rpy2


class RAPI:
    _instance = None

    @clasmethod
    def get(cls):
        if cls._instance is None:
            rpy2 = import_r()
            cls._instance = cls(rpy2.rinterface, rpy2.robjects)
        return cls._instance

    def __init__(self, rinterface, robjects):
        self.rinterface = rinterface
        self.obj = robjects

    @contextmanger
    def capture_output(self, save_info):
        if save_info:
            # save output from the R console in output
            output = []
            callback = output.append
        else:
            output = None
            callback = lambda line: None

        self.rinterface.set_writeconsole_regular(callback)
        self.rinterface.set_writeconsole_warnerror(callback)

        try:
            yield output
        finally:
            self.rinterface.set_writeconsole_regular(
                self.rinterface.consolePrint
            )
            self.rinterface.set_writeconsole_warnerror(
                self.rinterface.consolePrint
            )
