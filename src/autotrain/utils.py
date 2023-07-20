import glob
import json
import os
import re
import shutil
import subprocess
import traceback
from typing import Dict, Optional

import requests
from huggingface_hub import HfApi, HfFolder
from huggingface_hub.repository import Repository
from loguru import logger
from transformers import AutoConfig

from autotrain import config
from autotrain.tasks import TASKS


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


def get_auth_headers(token: str, prefix: str = "Bearer"):
    return {"Authorization": f"{prefix} {token}"}


def http_get(
    path: str,
    token: str,
    domain: str = config.AUTOTRAIN_BACKEND_API,
    token_prefix: str = "Bearer",
    suppress_logs: bool = False,
    **kwargs,
) -> requests.Response:
    """HTTP GET request to the AutoNLP API, raises UnreachableAPIError if the API cannot be reached"""
    logger.info(f"Sending GET request to {domain + path}")
    try:
        response = requests.get(
            url=domain + path, headers=get_auth_headers(token=token, prefix=token_prefix), **kwargs
        )
    except requests.exceptions.ConnectionError:
        raise UnreachableAPIError("❌ Failed to reach AutoNLP API, check your internet connection")
    response.raise_for_status()
    return response


def http_post(
    path: str,
    token: str,
    payload: Optional[Dict] = None,
    domain: str = config.AUTOTRAIN_BACKEND_API,
    suppress_logs: bool = False,
    **kwargs,
) -> requests.Response:
    """HTTP POST request to the AutoNLP API, raises UnreachableAPIError if the API cannot be reached"""
    logger.info(f"Sending POST request to {domain + path}")
    try:
        response = requests.post(
            url=domain + path, json=payload, headers=get_auth_headers(token=token), allow_redirects=True, **kwargs
        )
    except requests.exceptions.ConnectionError:
        raise UnreachableAPIError("❌ Failed to reach AutoNLP API, check your internet connection")
    response.raise_for_status()
    return response


def get_task(task_id: int) -> str:
    for key, value in TASKS.items():
        if value == task_id:
            return key
    return "❌ Unsupported task! Please update autonlp"


def get_user_token():
    return HfFolder.get_token()


def user_authentication(token):
    logger.info("Authenticating user...")
    headers = {}
    cookies = {}
    if token.startswith("hf_"):
        headers["Authorization"] = f"Bearer {token}"
    else:
        cookies = {"token": token}
    try:
        response = requests.get(
            config.HF_API + "/api/whoami-v2",
            headers=headers,
            cookies=cookies,
            timeout=3,
        )
    except (requests.Timeout, ConnectionError) as err:
        logger.error(f"Failed to request whoami-v2 - {repr(err)}")
        raise Exception("Hugging Face Hub is unreachable, please try again later.")
    return response.json()


def get_project_cost(username, token, task, num_samples, num_models):
    logger.info("Getting project cost...")
    task_id = TASKS[task]
    pricing = http_get(
        path=f"/pricing/compute?username={username}&task_id={task_id}&num_samples={num_samples}&num_models={num_models}",
        token=token,
    )
    return pricing.json()["price"]


def app_error_handler(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as err:
            logger.error(f"{func.__name__} has failed due to an exception:")
            logger.error(traceback.format_exc())
            if "param_choice" in str(err):
                ValueError("Unable to estimate costs. Job params not chosen yet.")
            elif "Failed to reach AutoNLP API" in str(err):
                ValueError("Unable to reach AutoTrain API. Please check your internet connection.")
            elif "An error has occurred: 'NoneType' object has no attribute 'type'" in str(err):
                ValueError("Unable to estimate costs. Data not uploaded yet.")
            else:
                ValueError(f"An error has occurred: {err}")

    return wrapper


def clone_hf_repo(repo_url: str, local_dir: str, token: str) -> Repository:
    os.makedirs(local_dir, exist_ok=True)
    repo_url = re.sub(r"(https?://)", rf"\1user:{token}@", repo_url)
    subprocess.run(
        "git lfs install".split(),
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
        check=True,
        encoding="utf-8",
        cwd=local_dir,
    )

    subprocess.run(
        f"git lfs clone {repo_url} .".split(),
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
        check=True,
        encoding="utf-8",
        cwd=local_dir,
    )

    data_repo = Repository(local_dir=local_dir, use_auth_token=token)
    return data_repo


def create_repo(project_name, autotrain_user, huggingface_token, model_path):
    repo_name = f"autotrain-{project_name}"
    repo_url = HfApi().create_repo(
        repo_id=f"{autotrain_user}/{repo_name}",
        token=huggingface_token,
        exist_ok=False,
        private=True,
    )
    if len(repo_url.strip()) == 0:
        repo_url = f"https://huggingface.co/{autotrain_user}/{repo_name}"

    logger.info(f"Created repo: {repo_url}")

    model_repo = clone_hf_repo(
        local_dir=model_path,
        repo_url=repo_url,
        token=huggingface_token,
    )
    model_repo.lfs_track(patterns=LFS_PATTERNS)
    return model_repo


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
