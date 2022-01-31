import warnings


def _suppress_useless_warnings() -> None:
    warnings.simplefilter("ignore", FutureWarning)


_suppress_useless_warnings()
