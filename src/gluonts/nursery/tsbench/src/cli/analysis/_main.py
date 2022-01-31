from cli._main import main


@main.group()  # type: ignore
def analysis():
    """
    Analyze evaluations locally and track via Sacred.
    """
