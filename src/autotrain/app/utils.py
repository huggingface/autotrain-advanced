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
    """
    Retrieves and manages running jobs from the database.

    This function fetches the list of running jobs from the provided database object.
    For each running job, it checks the process status. If the status is "completed",
    "error", or "zombie", it attempts to kill the process and remove the job from the
    database. After processing, it fetches and returns the updated list of running jobs.

    Args:
        db: A database object that provides methods to get and delete running jobs.

    Returns:
        list: An updated list of running jobs from the database.
    """
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
    """
    Retrieve the status of a process given its PID.

    Args:
        pid (int): The process ID of the process to check.

    Returns:
        str: The status of the process. If the process does not exist, returns "Completed".

    Raises:
        psutil.NoSuchProcess: If no process with the given PID is found.
    """
    try:
        process = psutil.Process(pid)
        proc_status = process.status()
        return proc_status
    except psutil.NoSuchProcess:
        logger.info(f"No process found with PID: {pid}")
        return "Completed"


def kill_process_by_pid(pid):
    """
    Kill a process by its PID (Process ID).

    This function attempts to terminate a process with the given PID using the SIGTERM signal.
    It logs the outcome of the operation, whether successful or not.

    Args:
        pid (int): The Process ID of the process to be terminated.

    Raises:
        ProcessLookupError: If no process with the given PID is found.
        Exception: If an error occurs while attempting to send the SIGTERM signal.
    """
    try:
        os.kill(pid, signal.SIGTERM)
        logger.info(f"Sent SIGTERM to process with PID {pid}")
    except ProcessLookupError:
        logger.error(f"No process found with PID {pid}")
    except Exception as e:
        logger.error(f"Failed to send SIGTERM to process with PID {pid}: {e}")


def token_verification(token):
    """
    Verifies the provided token with the Hugging Face API and retrieves user information.

    Args:
        token (str): The token to be verified. It can be either an OAuth token (starting with "hf_oauth")
                     or a regular token (starting with "hf_").

    Returns:
        dict: A dictionary containing user information with the following keys:
            - id (str): The user ID.
            - name (str): The user's preferred username.
            - orgs (list): A list of organizations the user belongs to.

    Raises:
        Exception: If the Hugging Face Hub is unreachable or the token is invalid.
    """
    if token.startswith("hf_oauth"):
        _api_url = config.HF_API + "/oauth/userinfo"
        _err_msg = "/oauth/userinfo"
    else:
        _api_url = config.HF_API + "/api/whoami-v2"
        _err_msg = "/api/whoami-v2"
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
        logger.error(f"Failed to request {_err_msg} - {repr(err)}")
        raise Exception(f"Hugging Face Hub ({_err_msg}) is unreachable, please try again later.")

    if response.status_code != 200:
        logger.error(f"Failed to request {_err_msg} - {response.status_code}")
        raise Exception(f"Invalid token ({_err_msg}). Please login with a write token.")

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
    """
    Retrieve the username and organizations associated with the provided user token.

    Args:
        user_token (str): The token used to authenticate the user. Must be a valid write token.

    Returns:
        list: A list containing the username followed by the organizations the user belongs to.

    Raises:
        Exception: If the user token is None or an empty string.
    """
    if user_token is None:
        raise Exception("Please login with a write token.")

    if user_token is None or len(user_token) == 0:
        raise Exception("Invalid token. Please login with a write token.")

    user_info = token_verification(token=user_token)
    username = user_info["name"]
    orgs = user_info["orgs"]

    who_is_training = [username] + orgs

    return who_is_training
