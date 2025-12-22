import logging
from logging import Logger
from typing import Any

from transformers import TextStreamer


def logger_or_dummy(logger: Logger | None):
    if logger is None:
        logger = logging.getLogger("dummy_logger")
        logger.addHandler(logging.NullHandler())
    return logger


class DummyStreamer(TextStreamer):
    def __init__(self) -> None:
        pass

    def put(self, value: Any) -> None:
        pass

    def end(self) -> None:
        pass


def streamer_or_dummy(streamer: TextStreamer | None) -> TextStreamer:
    if streamer is None:
        return DummyStreamer()
    return streamer
