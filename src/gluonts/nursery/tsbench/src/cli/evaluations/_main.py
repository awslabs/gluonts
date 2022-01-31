from cli._main import main


@main.group()  # type: ignore
def evaluations():
    """
    Manage TSBench evaluations.
    """
