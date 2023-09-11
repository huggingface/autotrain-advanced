import json
import os
import subprocess

import psutil
from fastapi import FastAPI

from autotrain import logger
from autotrain.trainers.clm.params import LLMTrainingParams
from autotrain.trainers.dreambooth.params import DreamBoothTrainingParams
from autotrain.trainers.generic.params import GenericParams
from autotrain.trainers.tabular.params import TabularParams
from autotrain.trainers.text_classification.params import TextClassificationParams


HF_TOKEN = os.environ.get("HF_TOKEN")
AUTOTRAIN_USERNAME = os.environ.get("AUTOTRAIN_USERNAME")
PROJECT_NAME = os.environ.get("PROJECT_NAME")
TASK_ID = int(os.environ.get("TASK_ID"))
PARAMS = os.environ.get("PARAMS")
DATA_PATH = os.environ.get("DATA_PATH")
MODEL = os.environ.get("MODEL")
OUTPUT_MODEL_REPO = os.environ.get("OUTPUT_MODEL_REPO")
PID = None


api = FastAPI()
logger.info(f"AUTOTRAIN_USERNAME: {AUTOTRAIN_USERNAME}")
logger.info(f"PROJECT_NAME: {PROJECT_NAME}")
logger.info(f"TASK_ID: {TASK_ID}")
logger.info(f"DATA_PATH: {DATA_PATH}")
logger.info(f"MODEL: {MODEL}")
logger.info(f"OUTPUT_MODEL_REPO: {OUTPUT_MODEL_REPO}")


def run_training():
    params = json.loads(PARAMS)
    logger.info(params)
    if TASK_ID == 9:
        params = LLMTrainingParams.parse_raw(params)
        params.project_name = "/tmp/model"
        params.save(output_dir=params.project_name)
        cmd = ["accelerate", "launch", "--num_machines", "1", "--num_processes", "1"]
        cmd.append("--mixed_precision")
        if params.fp16:
            cmd.append("fp16")
        else:
            cmd.append("no")

        cmd.extend(
            [
                "-m",
                "autotrain.trainers.clm",
                "--training_config",
                os.path.join(params.project_name, "training_params.json"),
            ]
        )
    elif TASK_ID in (1, 2):
        params = TextClassificationParams.parse_raw(params)
        params.project_name = "/tmp/model"
        params.save(output_dir=params.project_name)
        cmd = ["accelerate", "launch", "--num_machines", "1", "--num_processes", "1"]
        cmd.append("--mixed_precision")
        if params.fp16:
            cmd.append("fp16")
        else:
            cmd.append("no")

        cmd.extend(
            [
                "-m",
                "autotrain.trainers.text_classification",
                "--training_config",
                os.path.join(params.project_name, "training_params.json"),
            ]
        )
    elif TASK_ID in (13, 14, 15, 16, 26):
        params = TabularParams.parse_raw(params)
        params.project_name = "/tmp/model"
        params.save(output_dir=params.project_name)
        cmd = [
            "python",
            "-m",
            "autotrain.trainers.tabular",
            "--training_config",
            os.path.join(params.project_name, "training_params.json"),
        ]
    elif TASK_ID == 27:
        params = GenericParams.parse_raw(params)
        params.project_name = "/tmp/model"
        params.save(output_dir=params.project_name)
        cmd = [
            "python",
            "-m",
            "autotrain.trainers.generic",
            "--config",
            os.path.join(params.project_name, "training_params.json"),
        ]
    elif TASK_ID == 25:
        params = DreamBoothTrainingParams.parse_raw(params)
        params.project_name = "/tmp/model"
        params.save(output_dir=params.project_name)
        cmd = [
            "python",
            "-m",
            "autotrain.trainers.dreambooth",
            "--training_config",
            os.path.join(params.project_name, "training_params.json"),
        ]

    else:
        raise NotImplementedError

    cmd = [str(c) for c in cmd]
    logger.info(cmd)
    env = os.environ.copy()
    process = subprocess.Popen(" ".join(cmd), shell=True, env=env)
    return process.pid


def get_process_status(pid):
    try:
        process = psutil.Process(pid)
        return process.status()
    except psutil.NoSuchProcess:
        return "No process found with PID: {}".format(pid)


def kill_process(pid):
    try:
        parent_process = psutil.Process(pid)
        children = parent_process.children(recursive=True)  # This will get all the child processes recursively

        # First, terminate the child processes
        for child in children:
            child.terminate()

        # Wait for the child processes to terminate, and kill them if they don't
        gone, still_alive = psutil.wait_procs(children, timeout=3)
        for child in still_alive:
            child.kill()

        # Now, terminate the parent process
        parent_process.terminate()
        parent_process.wait(timeout=5)

        logger.info(f"Process with pid {pid} and its children have been killed")
        return f"Process with pid {pid} and its children have been killed"

    except psutil.NoSuchProcess:
        logger.info(f"No process found with pid {pid}")
        return f"No process found with pid {pid}"

    except psutil.TimeoutExpired:
        logger.info(f"Process {pid} or one of its children has not terminated in time")
        return f"Process {pid} or one of its children has not terminated in time"


@api.on_event("startup")
async def startup_event():
    process_pid = run_training()
    logger.info(f"Started training with PID {process_pid}")
    global PID
    PID = process_pid


@api.get("/")
async def root():
    return "Your model is being trained..."


@api.get("/status")
async def status():
    return get_process_status(pid=PID)


@api.get("/kill")
async def kill():
    return kill_process(pid=PID)


@api.get("/health")
async def health():
    return "OK"
