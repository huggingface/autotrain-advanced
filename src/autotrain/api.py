import json
import os
import subprocess

import psutil
from fastapi import FastAPI

from autotrain import logger


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


def run_training():
    params = json.loads(PARAMS)
    if TASK_ID in [1, 2]:
        cmd = [
            "autotrain",
            "text-classification",
            "--train",
            "--project-name",
            "output",
            "--data-path",
            DATA_PATH,
            "--model",
            MODEL,
            "--train-split",
            "train",
            "--valid-split",
            "validation",
            "--text-column",
            "autotrain_text",
            "--target-column",
            "autotrain_label",
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
            "--max-seq-length",
            params["max_seq_length"],
            "--lr",
            params["learning_rate"],
        ]
        if "max_grad_norm" in params:
            cmd.extend(["--max-grad-norm", params["max_grad_norm"]])
        if "seed" in params:
            cmd.extend(["--seed", params["seed"]])
        if "logging_steps" in params:
            cmd.extend(["--logging-steps", params["logging_steps"]])
        if "evaluation_strategy" in params:
            cmd.extend(["--evaluation-strategy", params["evaluation_strategy"]])
        if "save_total_limit" in params:
            cmd.extend(["--save-total-limit", params["save_total_limit"]])
        if "save_strategy" in params:
            cmd.extend(["--save-strategy", params["save_strategy"]])
        if "auto_find_batch_size" in params:
            cmd.extend(["--auto-find-batch-size", params["auto_find_batch_size"]])
        if "fp16" in params:
            cmd.extend(["--fp16"])

        cmd.extend(["--push-to-hub"])
        cmd.extend(["--repo-id", OUTPUT_MODEL_REPO])
    else:
        raise NotImplementedError

    cmd = [str(c) for c in cmd]
    logger.info(cmd)
    process = subprocess.Popen(" ".join(cmd), shell=True)
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
