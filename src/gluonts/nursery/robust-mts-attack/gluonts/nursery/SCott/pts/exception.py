def assert_pts(condition: bool, message: str, *args, **kwargs) -> None:
    if not condition:
        raise Exception(message.format(*args, **kwargs))
