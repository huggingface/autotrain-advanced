import os
import subprocess

import psutil
from fastapi import FastAPI
from loguru import logger

HF_TOKEN = os.environ.get("HF_TOKEN")
REPO_ID = os.environ.get("REPO_ID")
# TASK_ID = int(os.environ.get("TASK_ID"))
PID = None


def run_training():
    command = ["sleep", "100"]
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
