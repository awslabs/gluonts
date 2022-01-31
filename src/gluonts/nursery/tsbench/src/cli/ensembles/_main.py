from cli._main import main


@main.group()  # type: ignore
def ensembles() -> None:
    """
    Simulate ensembles of models tracked via TSBench.
    """
