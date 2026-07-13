
import logging
from contextlib import contextmanager


@contextmanager
def temporary_log_level(new_level):
    """
    Temporarily set the logging level to `new_level` and restore the original level afterwards.
    """
    logger = logging.getLogger()
    original_level = logger.level  # Save the current logging level
    logger.setLevel(new_level)       # Set to the new level
    try:
        yield
    finally:
        logger.setLevel(original_level)  # Restore the original level
