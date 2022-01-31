import boto3

_SESSION = None


def default_session() -> boto3.Session:
    """
    Returns the shared session to be used.
    """
    global _SESSION  # pylint: disable=global-statement
    if _SESSION is None:
        _SESSION = boto3.Session()  # type: ignore
    return _SESSION


def account_id() -> boto3.Session:
    """
    Returns the ID of the account.
    """
    return default_session().client("sts").get_caller_identity().get("Account")
