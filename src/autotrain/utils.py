import json
import os
import signal
import subprocess

import psutil
import requests

from autotrain import config, logger
from autotrain.commands import launch_command
from autotrain.trainers.clm.params import LLMTrainingParams
from autotrain.trainers.dreambooth.params import DreamBoothTrainingParams
from autotrain.trainers.generic.params import GenericParams
from autotrain.trainers.image_classification.params import ImageClassificationParams
from autotrain.trainers.seq2seq.params import Seq2SeqParams
from autotrain.trainers.tabular.params import TabularParams
from autotrain.trainers.text_classification.params import TextClassificationParams
from autotrain.trainers.text_regression.params import TextRegressionParams
from autotrain.trainers.token_classification.params import TokenClassificationParams


FORMAT_TAG = "\033[{code}m"
RESET_TAG = FORMAT_TAG.format(code=0)
BOLD_TAG = FORMAT_TAG.format(code=1)
RED_TAG = FORMAT_TAG.format(code=91)
GREEN_TAG = FORMAT_TAG.format(code=92)
YELLOW_TAG = FORMAT_TAG.format(code=93)
PURPLE_TAG = FORMAT_TAG.format(code=95)
CYAN_TAG = FORMAT_TAG.format(code=96)

ALLOW_REMOTE_CODE = os.environ.get("ALLOW_REMOTE_CODE", "true").lower() == "true"

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


def get_running_jobs(db):
    running_jobs = db.get_running_jobs()
    if running_jobs:
        for _pid in running_jobs:
            proc_status = get_process_status(_pid)
            proc_status = proc_status.strip().lower()
            if proc_status in ("completed", "error", "zombie"):
                logger.info(f"Killing PID: {_pid}")
                try:
                    kill_process_by_pid(_pid)
                except Exception as e:
                    logger.info(f"Error while killing process: {e}")
                    logger.info(f"Process {_pid} is already completed. Skipping...")
                db.delete_job(_pid)

    running_jobs = db.get_running_jobs()
    return running_jobs


def get_process_status(pid):
    try:
        process = psutil.Process(pid)
        proc_status = process.status()
        return proc_status
    except psutil.NoSuchProcess:
        logger.info(f"No process found with PID: {pid}")
        return "Completed"


def kill_process_by_pid(pid):
    """Kill process by PID."""
    os.kill(pid, signal.SIGTERM)


def user_authentication(token):
    if token.startswith("hf_oauth"):
        _api_url = config.HF_API + "/oauth/userinfo"
    else:
        _api_url = config.HF_API + "/api/whoami-v2"
    headers = {}
    cookies = {}
    if token.startswith("hf_"):
        headers["Authorization"] = f"Bearer {token}"
    else:
        cookies = {"token": token}
    try:
        response = requests.get(
            _api_url,
            headers=headers,
            cookies=cookies,
            timeout=3,
        )
    except (requests.Timeout, ConnectionError) as err:
        logger.error(f"Failed to request whoami-v2 - {repr(err)}")
        raise Exception("Hugging Face Hub is unreachable, please try again later.")
    resp = response.json()
    user_info = {}
    if "error" in resp:
        return resp
    if token.startswith("hf_oauth"):
        user_info["id"] = resp["sub"]
        user_info["name"] = resp["preferred_username"]
        user_info["orgs"] = [resp["orgs"][k]["preferred_username"] for k in range(len(resp["orgs"]))]
    else:
        user_info["id"] = resp["id"]
        user_info["name"] = resp["name"]
        user_info["orgs"] = [resp["orgs"][k]["name"] for k in range(len(resp["orgs"]))]
    return user_info


def user_validation(user_token):
    if user_token is None:
        raise Exception("Please login with a write token.")

    if user_token is None or len(user_token) == 0:
        raise Exception("Invalid token. Please login with a write token.")

    user_info = user_authentication(token=user_token)
    username = user_info["name"]
    orgs = user_info["orgs"]

    who_is_training = [username] + orgs

    return who_is_training


def run_training(params, task_id, local=False, wait=False):
    params = json.loads(params)
    if isinstance(params, str):
        params = json.loads(params)
    if task_id == 9:
        params = LLMTrainingParams(**params)
    elif task_id == 28:
        params = Seq2SeqParams(**params)
    elif task_id in (1, 2):
        params = TextClassificationParams(**params)
    elif task_id in (13, 14, 15, 16, 26):
        params = TabularParams(**params)
    elif task_id == 27:
        params = GenericParams(**params)
    elif task_id == 25:
        params = DreamBoothTrainingParams(**params)
    elif task_id == 18:
        params = ImageClassificationParams(**params)
    elif task_id == 4:
        params = TokenClassificationParams(**params)
    elif task_id == 10:
        params = TextRegressionParams(**params)
    else:
        raise NotImplementedError

    params.save(output_dir=params.project_name)
    cmd = launch_command(params=params)
    cmd = [str(c) for c in cmd]
    env = os.environ.copy()
    process = subprocess.Popen(cmd, env=env)
    if wait:
        process.wait()
    return process.pid
