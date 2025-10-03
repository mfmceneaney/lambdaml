# LOG
import logging
import sys
import pytest

# Local imports
from lambdaml.log import setup_logger, set_global_log_level


@pytest.fixture(name="levels")
def levels_fixture():
    return ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


def test_setup_logger(levels):
    logger = setup_logger("test_logger")
    for level in levels:
        set_global_log_level(level)
        assert logger.getEffectiveLevel() == getattr(logging, level.upper())
    assert logger.name == "test_logger"
    assert len(logger.handlers) == 1
    assert isinstance(logger.handlers[0], logging.StreamHandler)
    assert logger.handlers[0].formatter._fmt == "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
    assert logger.handlers[0].formatter.datefmt == "%H:%M:%S"


def test_set_global_log_level(levels):
    logger = setup_logger("test_logger")
    for level in levels:
        set_global_log_level(level)
        assert logger.getEffectiveLevel() == getattr(logging, level.upper())
