# LOG
import logging
import sys


def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s", datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False  # <- Prevent messages from bubbling to root

    return logger


def set_global_log_level(level_str: str):
    level = getattr(logging, level_str.upper(), logging.INFO)

    # Update all existing loggers
    logger_dict = logging.Logger.manager.loggerDict
    for _, logger in logger_dict.items():
        if isinstance(logger, logging.Logger):
            logger.setLevel(level)
