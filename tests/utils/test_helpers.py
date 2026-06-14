import importlib
import logging
import sys


def test_get_logger_uses_log_level_env(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    sys.modules.pop("model_exporter.utils.helpers", None)

    helpers = importlib.import_module("model_exporter.utils.helpers")
    logger = helpers.get_logger("test-log-level-env")

    try:
        assert logger.level == logging.DEBUG
        assert all(handler.level == logging.DEBUG for handler in logger.handlers)
    finally:
        logger.handlers.clear()


def test_default_opset_uses_function_default(monkeypatch):
    sys.modules.pop("model_exporter.utils.helpers", None)

    helpers = importlib.import_module("model_exporter.utils.helpers")

    assert helpers.get_default_opset() == 17
    assert helpers.get_default_opset(default=18) == 18
