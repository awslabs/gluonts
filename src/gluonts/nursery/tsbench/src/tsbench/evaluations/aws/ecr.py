from .session import account_id, default_session


def image_uri(path: str) -> str:
    """
    Returns the ECR image URI for the model at the specified path.

    Args:
        path: The path, including the tag.

    Returns:
        The image URI.
    """
    return f"{account_id()}.dkr.ecr.{default_session().region_name}.amazonaws.com/{path}"
