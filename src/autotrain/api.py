import asyncio
import os
import signal
from contextlib import asynccontextmanager

from fastapi import FastAPI

from autotrain import logger
from autotrain.app_utils import get_running_jobs, run_training
from autotrain.db import AutoTrainDB


log_file_path = "/tmp/app.log"

HF_TOKEN = os.environ.get("HF_TOKEN")
AUTOTRAIN_USERNAME = os.environ.get("AUTOTRAIN_USERNAME")
PROJECT_NAME = os.environ.get("PROJECT_NAME")
TASK_ID = int(os.environ.get("TASK_ID"))
PARAMS = os.environ.get("PARAMS")
DATA_PATH = os.environ.get("DATA_PATH")
MODEL = os.environ.get("MODEL")
OUTPUT_MODEL_REPO = os.environ.get("OUTPUT_MODEL_REPO")
DB = AutoTrainDB("autotrain.db")


class BackgroundRunner:
    async def run_main(self):
        while True:
            running_jobs = get_running_jobs(DB)
            if not running_jobs:
                logger.info("No running jobs found. Shutting down the server.")
                os.kill(os.getpid(), signal.SIGINT)
            else:
                logger.info("No running jobs found.")
            await asyncio.sleep(30)


runner = BackgroundRunner()


@asynccontextmanager
async def lifespan(app: FastAPI):
    process_pid = run_training(params=PARAMS, task_id=TASK_ID)
    logger.info(f"Started training with PID {process_pid}")
    DB.add_job(process_pid)
    asyncio.create_task(runner.run_main())
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


@api.get("/health")
async def health():
    return "OK"
