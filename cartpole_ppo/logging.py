import sys
from loguru import logger

# remove the default logger
logger.remove()

# set the default logger to stderr:
logger.add(
    sys.stderr,
    format="{time} {level} {message}",
    level="DEBUG",
)

# set a file handler:
logger.add(
    "logs/logfile_{time}.log",
    rotation="100 MB",
    compression="zip",
    level="DEBUG",
)

# expose the logger
__all__ = ["logger"]
