import glob
import json
import os
import shutil
import traceback

from transformers import AutoConfig

from autotrain import logger


FORMAT_TAG = "\033[{code}m"
RESET_TAG = FORMAT_TAG.format(code=0)
BOLD_TAG = FORMAT_TAG.format(code=1)
RED_TAG = FORMAT_TAG.format(code=91)
GREEN_TAG = FORMAT_TAG.format(code=92)
YELLOW_TAG = FORMAT_TAG.format(code=93)
PURPLE_TAG = FORMAT_TAG.format(code=95)
CYAN_TAG = FORMAT_TAG.format(code=96)

LFS_PATTERNS = [
    "*.bin.*",
    "*.lfs.*",
    "*.bin",
    "*.h5",
    "*.tflite",
    "*.tar.gz",
    "*.ot",
    "*.onnx",
    "*.pt",
    "*.pkl",
    "*.parquet",
    "*.joblib",
    "tokenizer.json",
]


class UnauthenticatedError(Exception):
    pass


class UnreachableAPIError(Exception):
    pass


def save_model(torch_model, model_path):
    torch_model.save_pretrained(model_path)
    try:
        torch_model.save_pretrained(model_path, safe_serialization=True)
    except Exception as e:
        logger.error(f"Safe serialization failed with error: {e}")


def save_tokenizer(tok, model_path):
    tok.save_pretrained(model_path)


def update_model_config(model, job_config):
    model.config._name_or_path = "AutoTrain"
    if job_config.task in ("speech_recognition", "summarization"):
        return model
    if "max_seq_length" in job_config:
        model.config.max_length = job_config.max_seq_length
        model.config.padding = "max_length"
    return model


def save_model_card(model_card, model_path):
    with open(os.path.join(model_path, "README.md"), "w") as fp:
        fp.write(f"{model_card}")


def create_file(filename, file_content, model_path):
    with open(os.path.join(model_path, filename), "w") as fp:
        fp.write(f"{file_content}")


def save_config(conf, model_path):
    with open(os.path.join(model_path, "config.json"), "w") as fp:
        json.dump(conf, fp)


def remove_checkpoints(model_path):
    subfolders = glob.glob(os.path.join(model_path, "*/"))
    for subfolder in subfolders:
        shutil.rmtree(subfolder)
    try:
        os.remove(os.path.join(model_path, "emissions.csv"))
    except OSError:
        pass


def job_watcher(func):
    def wrapper(co2_tracker, *args, **kwargs):
        try:
            return func(co2_tracker, *args, **kwargs)
        except Exception:
            logger.error(f"{func.__name__} has failed due to an exception:")
            logger.error(traceback.format_exc())
            co2_tracker.stop()
            # delete training tracker file
            os.remove(os.path.join("/tmp", "training"))

    return wrapper


def get_model_architecture(model_path_or_name: str, revision: str = "main") -> str:
    config = AutoConfig.from_pretrained(model_path_or_name, revision=revision, trust_remote_code=True)
    architectures = config.architectures
    if architectures is None or len(architectures) > 1:
        raise ValueError(
            f"The model architecture is either not defined or not unique. Found architectures: {architectures}"
        )
    return architectures[0]
