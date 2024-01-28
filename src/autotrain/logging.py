import sys

from accelerate.state import PartialState
from loguru import logger


log_file_path = "/tmp/app.log"

emojis = {
    "TRACE": "🔍",
    "DEBUG": "🐛",
    "INFO": "🚀",
    "SUCCESS": "✅",
    "WARNING": "⚠️",
    "ERROR": "❌",
    "CRITICAL": "🚨",
}


def should_log(record):
    return PartialState().is_main_process


def emoji_filter(record):
    level = record["level"].name
    record["level_emoji"] = emojis.get(level, "") + " " + level
    return True


log_format = (
    "<level>{level_emoji: <8}</level> | "
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)

logger.remove()
if not hasattr(logger, "_is_customized") or not logger._is_customized:
    logger.add(sys.stderr, format=log_format, filter=lambda x: should_log(x) and emoji_filter(x))
    logger.add(log_file_path, format=log_format, filter=lambda x: should_log(x) and emoji_filter(x), rotation="10 MB")
    logger._is_customized = True

custom_logger = logger
