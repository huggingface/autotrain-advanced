import os
import sys

from loguru import logger


HF_AUTONLP_BACKEND_API = os.getenv("HF_AUTONLP_BACKEND_API", "https://api.autonlp.huggingface.co")

HF_API = os.getenv("HF_API", "https://huggingface.co/api")

logger.configure(handlers=[dict(sink=sys.stderr, format="> <level>{level:<7} {message}</level>")])
