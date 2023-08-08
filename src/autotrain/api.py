from fastapi import FastAPI, BackgroundTasks
import os
import time
from loguru import logger

HF_TOKEN = os.environ.get("HF_TOKEN")
REPO_ID = os.environ.get("REPO_ID")
# TASK_ID = int(os.environ.get("TASK_ID"))


def run_training():
    # os.makedirs("/tmp/output", exist_ok=True)
    for i in range(50):
        logger.info(f"Training {i}...")
        time.sleep(1)
        # save a temporary file
        # with open(f"/tmp/output/{i}.txt", "w") as f:
        #     f.write(f"Training {i}...")


api = FastAPI()


@api.on_event("startup")
async def startup_event():
    run_training()


@api.get("/")
async def root():
    return {"message": "Hello World"}
