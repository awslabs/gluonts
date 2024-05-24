import pandas as pd
from packaging.version import Version

if Version(pd.__version__) <= Version("2.2.0"):
    S = "S"
    H = "H"
    M = "M"
    Q = "Q"
    Y = "A"
else:
    S = "s"
    H = "h"
    M = "ME"
    Q = "QE"
    Y = "YE"
