import logging
import sys
import pytest

# Local imports
from lambdaml.log import setup_logger, set_global_log_level


def test_setup_logger():
    logger = setup_logger("test_logger")
    assert logger.name == "test_logger"
    print("logger.level = ",logger.level)
    print("logging.INFO = ",logging.INFO)
    assert logger.level == logging.INFO
    print("len(logger.handlers) = ",len(logger.handlers))
    assert len(logger.handlers) == 1
    assert isinstance(logger.handlers[0], logging.StreamHandler)
    assert logger.handlers[0].formatter._fmt == "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
    assert logger.handlers[0].formatter.datefmt == "%H:%M:%S"

def test_set_global_log_level():
    logger = setup_logger("test_logger")
    set_global_log_level("DEBUG")
    assert logger.level == logging.DEBUG
