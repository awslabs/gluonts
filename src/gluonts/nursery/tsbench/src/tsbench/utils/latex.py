import pandas as pd


def float_formatter(column: pd.Series, value: float, minimize: bool = True) -> str:  # type: ignore
    """
    Returns a formatter to be used when printing data frames to LaTeX.
    """
    if value == (column.min() if minimize else column.max()):
        return f"\\textbf{{{value:,.2f}}}"
    return f"{value:,.2f}"
