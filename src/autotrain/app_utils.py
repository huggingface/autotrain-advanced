import json
import os
import signal
import socket
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


def get_process_status(pid):
    try:
        process = psutil.Process(pid)
        proc_status = process.status()
        return proc_status
    except psutil.NoSuchProcess:
        logger.info(f"No process found with PID: {pid}")
        return "Completed"


def find_pid_by_port(port):
    """Find PID by port number."""
    try:
        result = subprocess.run(["lsof", "-i", f":{port}", "-t"], capture_output=True, text=True, check=True)
        pids = result.stdout.strip().split("\n")
        return [int(pid) for pid in pids if pid.isdigit()]
    except subprocess.CalledProcessError:
        return []


def kill_process_by_pid(pid):
    """Kill process by PID."""
    os.kill(pid, signal.SIGTERM)


def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


def kill_process_on_port(port):
    os.system(f"fuser -k {port}/tcp")


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


def _login_user(user_token):
    user_info = user_authentication(token=user_token)
    username = user_info["name"]

    user_can_pay = user_info["canPay"]
    orgs = user_info["orgs"]

    valid_orgs = [org for org in orgs if org["canPay"] is True]
    valid_orgs = [org for org in valid_orgs if org["roleInOrg"] in ("admin", "write")]
    valid_orgs = [org["name"] for org in valid_orgs]

    valid_can_pay = [username] + valid_orgs if user_can_pay else valid_orgs
    who_is_training = [username] + [org["name"] for org in orgs]
    return user_token, valid_can_pay, who_is_training


def user_validation():
    user_token = os.environ.get("HF_TOKEN", None)

    if user_token is None:
        raise ValueError("Please login with a write token.")

    user_token, valid_can_pay, who_is_training = _login_user(user_token)

    if user_token is None or len(user_token) == 0:
        raise ValueError("Please login with a write token.")

    return user_token, valid_can_pay, who_is_training


def run_training(params, task_id, local=False, wait=False):
    params = json.loads(params)
    logger.info(params)
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
    else:
        raise NotImplementedError

    if not local:
        params.project_name = "/tmp/model"
    params.save(output_dir=params.project_name)
    cmd = launch_command(params=params)

    cmd = [str(c) for c in cmd]
    logger.info(cmd)
    env = os.environ.copy()
    process = subprocess.Popen(" ".join(cmd), shell=True, env=env)
    if wait:
        process.wait()
    return process.pid
