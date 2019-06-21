import os


"""
Maximum number of times a transformation can receive an input without returning an output.
This parameter is intended to catch infinite loops or inefficiencies, when transformations
never or rarely return something.
"""
GLUONTS_MAX_IDLE_TRANSFORMS = int(
    os.environ.get('GLUONTS_MAX_IDLE_TRANSFORMS', '100')
)
