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

    return logger


def set_global_log_level(level_str: str):
    level = getattr(logging, level_str.upper(), logging.INFO)
    logging.basicConfig(level=level)  # affects root logger only

    # Update all existing loggers
    logger_dict = logging.root.manager.loggerDict
    for name, logger in logger_dict.items():
        if isinstance(logger, logging.Logger):
            logger.setLevel(level)
