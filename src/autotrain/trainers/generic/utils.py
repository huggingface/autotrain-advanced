import os
import subprocess

import requests
from huggingface_hub import HfApi, snapshot_download

from autotrain import logger


def create_dataset_repo(username, project_name, script_path, token):
    """
    Creates a new dataset repository on Hugging Face and uploads the specified dataset.

    Args:
        username (str): The username of the Hugging Face account.
        project_name (str): The name of the project for which the dataset repository is being created.
        script_path (str): The local path to the dataset folder that needs to be uploaded.
        token (str): The authentication token for the Hugging Face API.

    Returns:
        str: The repository ID of the newly created dataset repository.
    """
    logger.info("Creating dataset repo...")
    api = HfApi(token=token)
    repo_id = f"{username}/autotrain-{project_name}"
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=True,
    )
    logger.info("Uploading dataset...")
    api.upload_folder(
        folder_path=script_path,
        repo_id=repo_id,
        repo_type="dataset",
    )
    logger.info("Dataset uploaded.")
    return repo_id


def pull_dataset_repo(params):
    """
    Downloads a dataset repository from Hugging Face Hub.

    Args:
        params (object): An object containing the following attributes:
            - data_path (str): The repository ID of the dataset.
            - project_name (str): The local directory where the dataset will be downloaded.
            - token (str): The authentication token for accessing the repository.

    Returns:
        None
    """
    snapshot_download(
        repo_id=params.data_path,
        local_dir=params.project_name,
        token=params.token,
        repo_type="dataset",
    )


def uninstall_requirements(params):
    """
    Uninstalls the requirements specified in the requirements.txt file of a given project.

    This function reads the requirements.txt file located in the project's directory,
    extracts the packages to be uninstalled, writes them to an uninstall.txt file,
    and then uses pip to uninstall those packages.

    Args:
        params (object): An object containing the project_name attribute, which specifies
                         the directory of the project.

    Returns:
        None
    """
    if os.path.exists(f"{params.project_name}/requirements.txt"):
        # read the requirements.txt
        uninstall_list = []
        with open(f"{params.project_name}/requirements.txt", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("-"):
                    uninstall_list.append(line[1:])

        # create an uninstall.txt
        with open(f"{params.project_name}/uninstall.txt", "w", encoding="utf-8") as f:
            for line in uninstall_list:
                f.write(line)

        pipe = subprocess.Popen(
            [
                "pip",
                "uninstall",
                "-r",
                "uninstall.txt",
                "-y",
            ],
            cwd=params.project_name,
        )
        pipe.wait()
        logger.info("Requirements uninstalled.")
        return


def install_requirements(params):
    """
    Installs the Python packages listed in the requirements.txt file located in the specified project directory.

    Args:
        params: An object containing the project_name attribute, which specifies the directory of the project.

    Behavior:
        - Checks if a requirements.txt file exists in the project directory.
        - Reads the requirements.txt file and filters out lines starting with a hyphen.
        - Rewrites the filtered requirements back to the requirements.txt file.
        - Uses subprocess to run the pip install command on the requirements.txt file.
        - Logs the installation status.

    Returns:
        None
    """
    # check if params.project_name has a requirements.txt
    if os.path.exists(f"{params.project_name}/requirements.txt"):
        # install the requirements using subprocess, wait for it to finish
        install_list = []

        with open(f"{params.project_name}/requirements.txt", "r", encoding="utf-8") as f:
            for line in f:
                if not line.startswith("-"):
                    install_list.append(line)

        with open(f"{params.project_name}/requirements.txt", "w", encoding="utf-8") as f:
            for line in install_list:
                f.write(line)

        pipe = subprocess.Popen(
            [
                "pip",
                "install",
                "-r",
                "requirements.txt",
            ],
            cwd=params.project_name,
        )
        pipe.wait()
        logger.info("Requirements installed.")
        return
    logger.info("No requirements.txt found. Skipping requirements installation.")
    return


def run_command(params):
    """
    Executes a Python script with optional arguments in a specified project directory.

    Args:
        params (object): An object containing the following attributes:
            - project_name (str): The name of the project directory where the script is located.
            - args (dict): A dictionary of arguments to pass to the script. Keys are argument names, and values are argument values.

    Raises:
        ValueError: If the script.py file is not found in the specified project directory.

    Returns:
        None
    """
    if os.path.exists(f"{params.project_name}/script.py"):
        cmd = ["python", "script.py"]
        if params.args:
            for arg in params.args:
                cmd.append(f"--{arg}")
                if params.args[arg] != "":
                    cmd.append(params.args[arg])
        pipe = subprocess.Popen(cmd, cwd=params.project_name)
        pipe.wait()
        logger.info("Command finished.")
        return
    raise ValueError("No script.py found.")


def pause_endpoint(params):
    """
    Pauses a specific endpoint using the Hugging Face API.

    This function retrieves the endpoint ID from the environment variables,
    extracts the username and project name from the endpoint ID, constructs
    the API URL, and sends a POST request to pause the endpoint.

    Args:
        params (object): An object containing the token attribute for authorization.

    Returns:
        dict: The JSON response from the API call.
    """
    endpoint_id = os.environ["ENDPOINT_ID"]
    username = endpoint_id.split("/")[0]
    project_name = endpoint_id.split("/")[1]
    api_url = f"https://api.endpoints.huggingface.cloud/v2/endpoint/{username}/{project_name}/pause"
    headers = {"Authorization": f"Bearer {params.token}"}
    r = requests.post(api_url, headers=headers, timeout=120)
    return r.json()
