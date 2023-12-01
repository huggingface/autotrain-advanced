import asyncio
import os
import signal
import subprocess
import time
from contextlib import asynccontextmanager

import psutil
from fastapi import FastAPI

from autotrain import logger
from autotrain.app_utils import run_training


HF_TOKEN = os.environ.get("HF_TOKEN")
AUTOTRAIN_USERNAME = os.environ.get("AUTOTRAIN_USERNAME")
PROJECT_NAME = os.environ.get("PROJECT_NAME")
TASK_ID = int(os.environ.get("TASK_ID"))
PARAMS = os.environ.get("PARAMS")
DATA_PATH = os.environ.get("DATA_PATH")
MODEL = os.environ.get("MODEL")
OUTPUT_MODEL_REPO = os.environ.get("OUTPUT_MODEL_REPO")
PID = None
API_PORT = os.environ.get("API_PORT", None)
logger.info(f"API_PORT: {API_PORT}")


class BackgroundRunner:
    async def run_main(self):
        while True:
            status = get_process_status(PID)
            status = status.strip().lower()
            if status in ("completed", "error", "zombie"):
                logger.info("Training process finished. Shutting down the server.")
                time.sleep(5)
                if API_PORT is not None:
                    subprocess.run(f"fuser -k {API_PORT}/tcp", shell=True, check=True)
                else:
                    kill_process(os.getpid())
                break
            time.sleep(5)


runner = BackgroundRunner()


def get_process_status(pid):
    try:
        process = psutil.Process(pid)
        proc_status = process.status()
        logger.info(f"Process status: {proc_status}")
        return proc_status
    except psutil.NoSuchProcess:
        logger.info(f"No process found with PID: {pid}")
        return "Completed"


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


def monitor_training_process(pid: int):
    status = get_process_status(pid)
    if status == "Completed" or status == "Error":
        logger.info("Training process finished. Shutting down the server.")
        os.kill(os.getpid(), signal.SIGINT)


@asynccontextmanager
async def lifespan(app: FastAPI):
    process_pid = run_training(params=PARAMS, task_id=TASK_ID)
    logger.info(f"Started training with PID {process_pid}")
    global PID
    PID = process_pid
    asyncio.create_task(runner.run_main())
    # background_tasks.add_task(monitor_training_process, PID)
    yield


api = FastAPI(lifespan=lifespan)
logger.info(f"AUTOTRAIN_USERNAME: {AUTOTRAIN_USERNAME}")
logger.info(f"PROJECT_NAME: {PROJECT_NAME}")
logger.info(f"TASK_ID: {TASK_ID}")
logger.info(f"DATA_PATH: {DATA_PATH}")
logger.info(f"MODEL: {MODEL}")
logger.info(f"OUTPUT_MODEL_REPO: {OUTPUT_MODEL_REPO}")


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
