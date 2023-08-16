import os
import sys

from autotrain import logger


AUTOTRAIN_BACKEND_API = os.getenv("AUTOTRAIN_BACKEND_API", "https://api.autotrain.huggingface.co")

HF_API = os.getenv("HF_API", "https://huggingface.co")


logger.configure(handlers=[dict(sink=sys.stderr, format="> <level>{level:<7} {message}</level>")])
