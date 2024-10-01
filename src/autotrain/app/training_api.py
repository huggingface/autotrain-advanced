import asyncio
import os
import signal
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI

from autotrain import logger
from autotrain.app.db import AutoTrainDB
from autotrain.app.utils import get_running_jobs, kill_process_by_pid
from autotrain.utils import run_training


HF_TOKEN = os.environ.get("HF_TOKEN")
AUTOTRAIN_USERNAME = os.environ.get("AUTOTRAIN_USERNAME")
PROJECT_NAME = os.environ.get("PROJECT_NAME")
TASK_ID = int(os.environ.get("TASK_ID"))
PARAMS = os.environ.get("PARAMS")
DATA_PATH = os.environ.get("DATA_PATH")
MODEL = os.environ.get("MODEL")
DB = AutoTrainDB("autotrain.db")


def graceful_exit(signum, frame):
    """
    Handles the SIGTERM signal to perform cleanup and exit the program gracefully.

    Args:
        signum (int): The signal number.
        frame (FrameType): The current stack frame (or None).

    Logs a message indicating that SIGTERM was received and then exits the program with status code 0.
    """
    logger.info("SIGTERM received. Performing cleanup...")
    sys.exit(0)


signal.signal(signal.SIGTERM, graceful_exit)


class BackgroundRunner:
    """
    A class to handle background running tasks.

    Methods
    -------
    run_main():
        Continuously checks for running jobs and shuts down the server if no jobs are found.
    """

    async def run_main(self):
        while True:
            running_jobs = get_running_jobs(DB)
            if not running_jobs:
                logger.info("No running jobs found. Shutting down the server.")
                kill_process_by_pid(os.getpid())
            await asyncio.sleep(30)


runner = BackgroundRunner()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the lifespan of the FastAPI application.

    This function is responsible for starting the training process and
    managing a background task runner. It logs the process ID of the
    training job, adds the job to the database, and ensures the background
    task is properly cancelled when the application shuts down.

    Args:
        app (FastAPI): The FastAPI application instance.

    Yields:
        None: This function is a generator that yields control back to the
        FastAPI application lifecycle.
    """
    process_pid = run_training(params=PARAMS, task_id=TASK_ID)
    logger.info(f"Started training with PID {process_pid}")
    DB.add_job(process_pid)
    task = asyncio.create_task(runner.run_main())
    yield

    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        logger.info("Background runner task cancelled.")


api = FastAPI(lifespan=lifespan)
logger.info(f"AUTOTRAIN_USERNAME: {AUTOTRAIN_USERNAME}")
logger.info(f"PROJECT_NAME: {PROJECT_NAME}")
logger.info(f"TASK_ID: {TASK_ID}")
logger.info(f"DATA_PATH: {DATA_PATH}")
logger.info(f"MODEL: {MODEL}")


@api.get("/")
async def root():
    return "Your model is being trained..."


@api.get("/health")
async def health():
    return "OK"
