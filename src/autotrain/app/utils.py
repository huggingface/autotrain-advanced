import os
import signal
import sys

import psutil
import requests

from autotrain import config, logger


def graceful_exit(signum, frame):
    logger.info("SIGTERM received. Performing cleanup...")
    sys.exit(0)


signal.signal(signal.SIGTERM, graceful_exit)


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
    try:
        os.kill(pid, signal.SIGTERM)
        logger.info(f"Sent SIGTERM to process with PID {pid}")
    except ProcessLookupError:
        logger.error(f"No process found with PID {pid}")
    except Exception as e:
        logger.error(f"Failed to send SIGTERM to process with PID {pid}: {e}")


def token_verification(token):
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

    if response.status_code != 200:
        logger.error(f"Failed to request whoami-v2 - {response.status_code}")
        raise Exception("Invalid token. Please login with a write token.")

    resp = response.json()
    user_info = {}

    if token.startswith("hf_oauth"):
        user_info["id"] = resp["sub"]
        user_info["name"] = resp["preferred_username"]
        user_info["orgs"] = [resp["orgs"][k]["preferred_username"] for k in range(len(resp["orgs"]))]
    else:
        user_info["id"] = resp["id"]
        user_info["name"] = resp["name"]
        user_info["orgs"] = [resp["orgs"][k]["name"] for k in range(len(resp["orgs"]))]
    return user_info


def get_user_and_orgs(user_token):
    if user_token is None:
        raise Exception("Please login with a write token.")

    if user_token is None or len(user_token) == 0:
        raise Exception("Invalid token. Please login with a write token.")

    user_info = token_verification(token=user_token)
    username = user_info["name"]
    orgs = user_info["orgs"]

    who_is_training = [username] + orgs

    return who_is_training
