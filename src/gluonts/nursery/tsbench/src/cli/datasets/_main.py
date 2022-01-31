from cli._main import main


@main.group()  # type: ignore
def datasets() -> None:
    """
    Manage datasets available in TSBench.
    """
