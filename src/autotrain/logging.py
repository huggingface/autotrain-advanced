import sys
from dataclasses import dataclass

from accelerate.state import PartialState
from loguru import logger


@dataclass
class Logger:
    def __post_init__(self):
        self.log_format = (
            "<level>{level: <8}</level> | "
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        )
        self.logger = logger
        self.setup_logger()

    def _should_log(self, record):
        return PartialState().is_main_process

    def setup_logger(self):
        self.logger.remove()
        self.logger.add(
            sys.stdout,
            format=self.log_format,
            filter=lambda x: self._should_log(x),
        )

    def get_logger(self):
        return self.logger
