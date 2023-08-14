import json
import os
import subprocess

import psutil
from fastapi import FastAPI
from loguru import logger


HF_TOKEN = os.environ.get("HF_TOKEN")
REPO_ID = os.environ.get("REPO_ID")
TASK_ID = int(os.environ.get("TASK_ID"))
PARAMS = os.environ.get("PARAMS")
DATA_PATH = os.environ.get("DATA_PATH")
MODEL = os.environ.get("MODEL")
DO_VALIDATION = int(os.environ.get("DO_VALIDATION", 0))
AUTOTRAIN_USERNAME = os.environ.get("AUTOTRAIN_USERNAME")
PROJECT_NAME = os.environ.get("PROJECT_NAME")
PID = None


def run_training():
    params = json.loads(PARAMS)
    output_repo = None
    if TASK_ID in [1, 2]:
        cmd = [
            "autotrain",
            "text-classification",
            "--data-path",
            DATA_PATH,
            "--model",
            MODEL,
            "--train-split",
            "train",
            "--valid-split",
            "valid",
            "--text-column",
            params["col_mapping_text"],
            "--target-column",
            params["col_mapping_target"],
            "--epochs",
            params["epochs"],
            "--batch-size",
            params["batch_size"],
            "--warmup-ratio",
            params["warmup_ratio"],
            "--gradient-accumulation",
            params["gradient_accumulation"],
            "--optimizer",
            params["optimizer"],
            "--scheduler",
            params["scheduler"],
            "--weight-decay",
            params["weight_decay"],
            "--max-grad-norm",
            params["max_grad_norm"],
            "--seed",
            params["seed"],
            "--logging-steps",
            params["logging_steps"],
            "--project-name",
            PROJECT_NAME,
            "--evaluation-strategy",
            params["evaluation_strategy"],
            "--save-total-limit",
            params["save_total_limit"],
            "--save-strategy",
            params["save_strategy"],
            "--auto-find-batch-size",
            params["auto_find_batch_size"],
            "--fp16",
            params["fp16"],
            "--push-to-hub",
            params["push_to_hub"],
            "--repo-id",
            f"{AUTOTRAIN_USERNAME}/{PROJECT_NAME}",
        ]
    process = subprocess.Popen(command, start_new_session=True)
    return process.pid


def get_process_status(pid):
    try:
        process = psutil.Process(pid)
        return process.status()
    except psutil.NoSuchProcess:
        return "No process found with PID: {}".format(pid)


def kill_process(pid):
    try:
        process = psutil.Process(pid)
        process.terminate()  # or process.kill()
        process.wait(timeout=5)
        logger.info(f"Process with pid {pid} has been killed")
        return "Process with pid {} has been killed".format(pid)
    except psutil.NoSuchProcess:
        logger.info(f"No process found with pid {pid}")
        return f"No process found with pid {pid}"
    except psutil.TimeoutExpired:
        logger.info(f"Process {pid} has not terminated in time")
        return f"Process {pid} has not terminated in time"


api = FastAPI()


@api.on_event("startup")
async def startup_event():
    process_pid = run_training()
    logger.info(f"Started training with PID {process_pid}")
    global PID
    PID = process_pid


@api.get("/")
async def root():
    return {"message": "Hello World"}


@api.get("/status")
async def status():
    return get_process_status(pid=PID)


@api.get("/kill")
async def kill():
    return kill_process(pid=PID)
