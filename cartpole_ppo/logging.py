from loguru import logger

# set a file handler:
logger.add(
    "logs/logfile_{time}.log",
    rotation="100 MB",
    level="DEBUG",
)
